#include "abjnnue_layers.h"
#include "abjnnue_runtime_layout.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(USE_AVX2)
    #include <immintrin.h>
#elif defined(USE_SSSE3)
    #include <tmmintrin.h>
#endif

namespace ABJNNUE::Layers {
namespace {

constexpr std::size_t L1BiasOffset      = RuntimeLayout::L1BiasOffset;
constexpr std::size_t L1WeightOffset    = RuntimeLayout::L1WeightOffset;
constexpr std::size_t L2BiasOffset      = RuntimeLayout::L2BiasOffset;
constexpr std::size_t L2WeightOffset    = RuntimeLayout::L2WeightOffset;
constexpr std::size_t FinalBiasOffset   = RuntimeLayout::FinalBiasOffset;
constexpr std::size_t FinalWeightOffset = RuntimeLayout::FinalWeightOffset;

static_assert(L1BiasOffset % 64 == 0 && L1WeightOffset % 64 == 0);
static_assert(L2BiasOffset % 64 == 0 && L2WeightOffset % 64 == 0);
static_assert(FinalBiasOffset % 64 == 0 && FinalWeightOffset % 64 == 0);
static_assert(L1BiasOffset + 32 * 4 == L1WeightOffset);
static_assert(L1WeightOffset + RuntimeLayout::HeadInputWidth * 32 == L2BiasOffset);
static_assert(L2BiasOffset + 32 * 4 == L2WeightOffset);
static_assert(L2WeightOffset + 64 * 32 == FinalBiasOffset);
static_assert(FinalWeightOffset + 128 == RuntimeLayout::EvalHeadBucketSize);

std::int32_t read_i32(const std::uint8_t* data) {
    const std::uint32_t value = std::uint32_t(data[0]) | (std::uint32_t(data[1]) << 8)
                              | (std::uint32_t(data[2]) << 16) | (std::uint32_t(data[3]) << 24);
    return static_cast<std::int32_t>(value);
}

template<std::size_t InputDimensions, std::size_t OutputDimensions, bool Sparse>
void affine_scalar(const std::uint8_t* input,
                   const std::uint8_t* biasBytes,
                   const std::uint8_t* weightBytes,
                   std::int32_t*       output) {
    static_assert(InputDimensions % 4 == 0);
    for (std::size_t out = 0; out < OutputDimensions; ++out)
        output[out] = read_i32(biasBytes + out * sizeof(std::int32_t));

    for (std::size_t chunk = 0; chunk < InputDimensions / 4; ++chunk)
    {
        std::uint32_t inputWord = 0;
        std::memcpy(&inputWord, input + chunk * 4, sizeof(inputWord));
        if constexpr (Sparse)
            if (inputWord == 0)
                continue;

        const auto* weights =
          reinterpret_cast<const std::int8_t*>(weightBytes + chunk * OutputDimensions * 4);
        for (std::size_t out = 0; out < OutputDimensions; ++out)
            for (std::size_t lane = 0; lane < 4; ++lane)
                output[out] += std::int32_t(weights[out * 4 + lane]) * input[chunk * 4 + lane];
    }
}

#if defined(USE_AVX2)

__m256i add_dpbusd(__m256i accumulator, __m256i input, __m256i weights) {
    auto products = _mm256_maddubs_epi16(input, weights);
    products      = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    return _mm256_add_epi32(accumulator, products);
}

template<std::size_t InputDimensions, std::size_t OutputDimensions, bool Sparse>
void affine_simd(const std::uint8_t* input,
                 const std::uint8_t* biasBytes,
                 const std::uint8_t* weightBytes,
                 std::int32_t*       output) {
    static_assert(InputDimensions % 4 == 0);
    static_assert(OutputDimensions % 8 == 0);
    constexpr std::size_t RegisterCount = OutputDimensions / 8;
    __m256i               accumulators[RegisterCount];
    for (std::size_t reg = 0; reg < RegisterCount; ++reg)
        accumulators[reg] =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(biasBytes + reg * sizeof(__m256i)));

    for (std::size_t chunk = 0; chunk < InputDimensions / 4; ++chunk)
    {
        std::uint32_t inputWord = 0;
        std::memcpy(&inputWord, input + chunk * 4, sizeof(inputWord));
        if constexpr (Sparse)
            if (inputWord == 0)
                continue;

        const auto  inputVector = _mm256_set1_epi32(static_cast<std::int32_t>(inputWord));
        const auto* weights     = weightBytes + chunk * OutputDimensions * 4;
        for (std::size_t reg = 0; reg < RegisterCount; ++reg)
        {
            const auto weightVector =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + reg * sizeof(__m256i)));
            accumulators[reg] = add_dpbusd(accumulators[reg], inputVector, weightVector);
        }
    }

    for (std::size_t reg = 0; reg < RegisterCount; ++reg)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + reg * 8), accumulators[reg]);
}

std::int32_t affine_final(const std::uint8_t* input,
                          const std::uint8_t* biasBytes,
                          const std::uint8_t* weightBytes) {
    auto sum = _mm256_setzero_si256();
    for (std::size_t offset = 0; offset < 128; offset += 32)
    {
        const auto inputVector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + offset));
        const auto weightVector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weightBytes + offset));
        sum = add_dpbusd(sum, inputVector, weightVector);
    }
    auto reduced = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    reduced      = _mm_add_epi32(reduced, _mm_shuffle_epi32(reduced, 0x4E));
    reduced      = _mm_add_epi32(reduced, _mm_shuffle_epi32(reduced, 0xB1));
    return _mm_cvtsi128_si32(reduced) + read_i32(biasBytes);
}

#elif defined(USE_SSSE3)

__m128i add_dpbusd(__m128i accumulator, __m128i input, __m128i weights) {
    auto products = _mm_maddubs_epi16(input, weights);
    products      = _mm_madd_epi16(products, _mm_set1_epi16(1));
    return _mm_add_epi32(accumulator, products);
}

template<std::size_t InputDimensions, std::size_t OutputDimensions, bool Sparse>
void affine_simd(const std::uint8_t* input,
                 const std::uint8_t* biasBytes,
                 const std::uint8_t* weightBytes,
                 std::int32_t*       output) {
    static_assert(InputDimensions % 4 == 0);
    static_assert(OutputDimensions % 4 == 0);
    constexpr std::size_t RegisterCount = OutputDimensions / 4;
    __m128i               accumulators[RegisterCount];
    for (std::size_t reg = 0; reg < RegisterCount; ++reg)
        accumulators[reg] =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(biasBytes + reg * sizeof(__m128i)));

    for (std::size_t chunk = 0; chunk < InputDimensions / 4; ++chunk)
    {
        std::uint32_t inputWord = 0;
        std::memcpy(&inputWord, input + chunk * 4, sizeof(inputWord));
        if constexpr (Sparse)
            if (inputWord == 0)
                continue;

        const auto  inputVector = _mm_set1_epi32(static_cast<std::int32_t>(inputWord));
        const auto* weights     = weightBytes + chunk * OutputDimensions * 4;
        for (std::size_t reg = 0; reg < RegisterCount; ++reg)
        {
            const auto weightVector =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + reg * sizeof(__m128i)));
            accumulators[reg] = add_dpbusd(accumulators[reg], inputVector, weightVector);
        }
    }

    for (std::size_t reg = 0; reg < RegisterCount; ++reg)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + reg * 4), accumulators[reg]);
}

std::int32_t affine_final(const std::uint8_t* input,
                          const std::uint8_t* biasBytes,
                          const std::uint8_t* weightBytes) {
    auto sum = _mm_setzero_si128();
    for (std::size_t offset = 0; offset < 128; offset += 16)
    {
        const auto inputVector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + offset));
        const auto weightVector =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(weightBytes + offset));
        sum = add_dpbusd(sum, inputVector, weightVector);
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
    return _mm_cvtsi128_si32(sum) + read_i32(biasBytes);
}

#endif

void sqr_clipped_relu(const std::int32_t* input, std::uint8_t* output, std::size_t count,
                      unsigned shift) {
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto clipped = std::clamp(input[i], -32768, 32767);
        const auto square  = std::int64_t(clipped) * clipped;
        output[i]          = static_cast<std::uint8_t>(std::min<std::int64_t>(127, square >> shift));
    }
}

void clipped_relu(const std::int32_t* input, std::uint8_t* output, std::size_t count,
                  unsigned shift) {
    for (std::size_t i = 0; i < count; ++i)
        output[i] = static_cast<std::uint8_t>(std::clamp(input[i] >> shift, 0, 127));
}

std::int32_t affine_final_scalar(const std::uint8_t* input,
                                 const std::uint8_t* biasBytes,
                                 const std::uint8_t* weightBytes) {
    std::int32_t output  = read_i32(biasBytes);
    const auto*  weights = reinterpret_cast<const std::int8_t*>(weightBytes);
    for (std::size_t i = 0; i < 128; ++i)
        output += std::int32_t(weights[i]) * input[i];
    return output;
}

std::int32_t propagate_impl(const std::uint8_t* head, const std::uint8_t* input, bool scalar) {
    alignas(32) std::int32_t first[32]{};
    if (scalar)
        affine_scalar<RuntimeLayout::HeadInputWidth, 32, true>(input, head + L1BiasOffset, head + L1WeightOffset, first);
    else
    {
#if defined(USE_AVX2) || defined(USE_SSSE3)
        affine_simd<RuntimeLayout::HeadInputWidth, 32, true>(input, head + L1BiasOffset, head + L1WeightOffset, first);
#else
        affine_scalar<RuntimeLayout::HeadInputWidth, 32, true>(input, head + L1BiasOffset, head + L1WeightOffset, first);
#endif
    }

    alignas(32) std::uint8_t squared[32]{};
    alignas(32) std::uint8_t clipped[32]{};
    sqr_clipped_relu(first, squared, 32, 21);
    clipped_relu(first, clipped, 32, 7);

    alignas(32) std::uint8_t secondInput[64]{};
    std::copy(squared, squared + 32, secondInput);
    std::copy(clipped, clipped + 32, secondInput + 32);

    alignas(32) std::int32_t second[32]{};
    if (scalar)
        affine_scalar<64, 32, false>(secondInput, head + L2BiasOffset, head + L2WeightOffset,
                                     second);
    else
    {
#if defined(USE_AVX2) || defined(USE_SSSE3)
        affine_simd<64, 32, false>(secondInput, head + L2BiasOffset, head + L2WeightOffset, second);
#else
        affine_scalar<64, 32, false>(secondInput, head + L2BiasOffset, head + L2WeightOffset,
                                     second);
#endif
    }

    alignas(32) std::uint8_t secondSquared[32]{};
    alignas(32) std::uint8_t secondClipped[32]{};
    sqr_clipped_relu(second, secondSquared, 32, 19);
    clipped_relu(second, secondClipped, 32, 6);
    alignas(32) std::uint8_t finalInput[128]{};
    std::copy(squared, squared + 32, finalInput);
    std::copy(clipped, clipped + 32, finalInput + 32);
    std::copy(secondSquared, secondSquared + 32, finalInput + 64);
    std::copy(secondClipped, secondClipped + 32, finalInput + 96);
    const auto final =
      scalar ? affine_final_scalar(finalInput, head + FinalBiasOffset, head + FinalWeightOffset)
#if defined(USE_AVX2) || defined(USE_SSSE3)
             : affine_final(finalInput, head + FinalBiasOffset, head + FinalWeightOffset);
#else
             : affine_final_scalar(finalInput, head + FinalBiasOffset, head + FinalWeightOffset);
#endif
    const auto skip = std::int64_t(first[30]) - first[31];
    return static_cast<std::int32_t>((std::int64_t(final) + skip) * (600 * 16) / (128 * 64 * 2));
}

}  // namespace

std::int32_t propagate(const std::uint8_t* head, const std::uint8_t* input) {
    return propagate_impl(head, input, false);
}

std::int32_t propagate_scalar(const std::uint8_t* head, const std::uint8_t* input) {
    return propagate_impl(head, input, true);
}

const char* backend_name() noexcept {
#if defined(USE_AVX2)
    return "avx2";
#elif defined(USE_SSSE3)
    return "ssse3";
#else
    return "scalar";
#endif
}

}  // namespace ABJNNUE::Layers
