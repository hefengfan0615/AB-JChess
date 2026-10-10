#include "abjnnue_inference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(USE_AVX2) || defined(USE_SSE2)
    #include <immintrin.h>
#endif

#include "../position.h"

namespace ABJNNUE {
namespace {

#if defined(ABJNNUE_RUNTIME_STATS)
thread_local InferenceStats runtimeStats;
#endif

void transform_perspective(const AccumulatorVector& values, std::uint8_t* output) {
#if defined(ABJNNUE_RUNTIME_STATS)
#    if defined(ABJNNUE_RUNTIME_SIMD_TRANSFORM) && defined(USE_SSE2)
    ++runtimeStats.simdTransforms;
#    else
    ++runtimeStats.scalarTransforms;
#    endif
#endif
#if defined(ABJNNUE_RUNTIME_SIMD_TRANSFORM) && defined(USE_AVX2)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i upper = _mm256_set1_epi16(127);
    for (std::size_t i = 0; i < RuntimeLayout::PerspectiveOutputWidth; i += 16)
    {
        auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data() + i));
        auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          values.data() + RuntimeLayout::PerspectiveOutputWidth + i));
        a = _mm256_min_epi16(_mm256_max_epi16(a, zero), upper);
        b = _mm256_min_epi16(_mm256_max_epi16(b, zero), upper);
        auto product = _mm256_mullo_epi16(a, b);
        product      = _mm256_srli_epi16(product, 7);
        const auto packed = _mm_packus_epi16(_mm256_castsi256_si128(product),
                                              _mm256_extracti128_si256(product, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), packed);
    }
#elif defined(ABJNNUE_RUNTIME_SIMD_TRANSFORM) && defined(USE_SSE2)
    const __m128i zero = _mm_setzero_si128();
    const __m128i upper = _mm_set1_epi16(127);
    for (std::size_t i = 0; i < RuntimeLayout::PerspectiveOutputWidth; i += 8)
    {
        auto a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values.data() + i));
        auto b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
          values.data() + RuntimeLayout::PerspectiveOutputWidth + i));
        a = _mm_min_epi16(_mm_max_epi16(a, zero), upper);
        b = _mm_min_epi16(_mm_max_epi16(b, zero), upper);
        auto product = _mm_mullo_epi16(a, b);
        product      = _mm_srli_epi16(product, 7);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), _mm_packus_epi16(product, product));
    }
#else
    for (std::size_t i = 0; i < RuntimeLayout::PerspectiveOutputWidth; ++i)
    {
        const auto a = std::clamp<std::int32_t>(values[i], 0, 127);
        const auto b = std::clamp<std::int32_t>(
          values[i + RuntimeLayout::PerspectiveOutputWidth], 0, 127);
        output[i] = static_cast<std::uint8_t>((a * b) >> 7);
    }
#endif
}

}  // namespace

const InferenceStats& Inference::stats() noexcept {
#if defined(ABJNNUE_RUNTIME_STATS)
    return runtimeStats;
#else
    static const InferenceStats empty{};
    return empty;
#endif
}

void Inference::reset_stats() noexcept {
#if defined(ABJNNUE_RUNTIME_STATS)
    runtimeStats = {};
#endif
}

RawEvaluation Inference::evaluate_accumulated(const Model& model,
                                              const Stockfish::Position& position,
                                              const AccumulatedPosition& accumulated,
                                              std::uint32_t layerStackBucket,
                                              TransformedFeatures* transformedOutput) {
    return evaluate_accumulated(model, position, accumulated,
                                LayerStackSelection{layerStackBucket, 0},
                                transformedOutput);
}

RawEvaluation Inference::evaluate_accumulated(const Model& model,
                                              const Stockfish::Position& position,
                                              const AccumulatedPosition& accumulated,
                                              LayerStackSelection selection,
                                              TransformedFeatures* transformedOutput) {
    return evaluate_accumulated(model, position, accumulated, selection,
                                FeatureEncoder::inventory_context(position,
                                                                   position.side_to_move()),
                                transformedOutput);
}

RawEvaluation Inference::evaluate_accumulated(const Model& model,
                                              const Stockfish::Position& position,
                                              const AccumulatedPosition& accumulated,
                                              LayerStackSelection selection,
                                              const InventoryContext& inventoryContext,
                                              TransformedFeatures* transformedOutput) {
    if (selection.floor >= RuntimeLayout::LayerStacks)
        throw std::logic_error("ABJNNUE layer-stack bucket is out of range");
    if (selection.floor == RuntimeLayout::LayerStacks - 1) selection.blendQ8 = 0;
    TransformedFeatures transformed{};
    const Stockfish::Color order[2] = {position.side_to_move(), ~position.side_to_move()};
    for (std::size_t perspectiveIndex = 0; perspectiveIndex < 2; ++perspectiveIndex)
    {
        const auto& values = accumulated.perspectives[order[perspectiveIndex]].values;
        transform_perspective(values,
                              transformed.data()
                                + perspectiveIndex * RuntimeLayout::PerspectiveOutputWidth);
    }
    RawEvaluation raw;
    raw.positionalRaw = model.propagate_interpolated(selection.floor,
                                                       selection.blendQ8,
                                                       transformed.data(),
                                                       inventoryContext);
    if (transformedOutput) *transformedOutput = transformed;
    return raw;
}

InferenceResult Inference::evaluate(const Model& model, const Stockfish::Position& position) {
    if (!model.inference_complete()) throw std::invalid_argument("ABJNNUE model is incomplete");
    InferenceResult result;
    result.encoded = FeatureEncoder::encode(position);
    result.accumulated = FeatureEncoder::accumulate(model, result.encoded);
    const auto raw = evaluate_accumulated(model, position, result.accumulated,
                                          result.encoded.layerStackSelection,
                                          &result.transformed);
    result.positionalRaw = raw.positionalRaw;
    return result;
}

}  // namespace ABJNNUE
