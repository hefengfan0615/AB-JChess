// Binary parity bridge: real feature rows and inventory context -> native heads.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "abjnnue/abjnnue_feature_encoder.h"
#include "abjnnue/abjnnue_inference.h"
#include "abjnnue/abjnnue_layers.h"
#include "bitboard.h"
#include "position.h"

template<class T> T read(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated parity input");
    return value;
}

static ABJNNUE::TransformedFeatures transform(const ABJNNUE::AccumulatedPosition& accumulated,
                                             std::uint32_t us) {
    ABJNNUE::TransformedFeatures result{};
    for (std::uint32_t side = 0; side < 2; ++side)
    {
        const auto color = side == 0 ? Stockfish::WHITE : Stockfish::BLACK;
        const auto& values = accumulated.perspectives[color].values;
        auto* output = result.data() + side * ABJNNUE::RuntimeLayout::PerspectiveOutputWidth;
        for (std::size_t i = 0; i < ABJNNUE::RuntimeLayout::PerspectiveOutputWidth; ++i)
        {
            const auto first = std::clamp<int>(values[i], 0, 127);
            const auto second = std::clamp<int>(values[i + ABJNNUE::RuntimeLayout::PerspectiveOutputWidth],
                                                0, 127);
            output[i] = static_cast<std::uint8_t>((first * second) >> 7);
        }
    }
    if (us == 0)
        return result;
    ABJNNUE::TransformedFeatures reordered{};
    std::copy_n(result.data() + ABJNNUE::RuntimeLayout::PerspectiveOutputWidth,
                ABJNNUE::RuntimeLayout::PerspectiveOutputWidth, reordered.data());
    std::copy_n(result.data(), ABJNNUE::RuntimeLayout::PerspectiveOutputWidth,
                reordered.data() + ABJNNUE::RuntimeLayout::PerspectiveOutputWidth);
    return reordered;
}

static std::array<std::uint8_t, ABJNNUE::RuntimeLayout::HeadInputWidth> make_head_input(
  const ABJNNUE::Model& model,
  const ABJNNUE::TransformedFeatures& transformed,
  const ABJNNUE::InventoryContext& context) {
    std::array<std::uint8_t, ABJNNUE::RuntimeLayout::HeadInputWidth> input{};
    std::copy(transformed.begin(), transformed.end(), input.begin());
    const auto& biases = model.context_biases();
    const auto& weights = model.context_weights();
    for (std::size_t output = 0; output < ABJNNUE::RuntimeLayout::InventoryContextHidden; ++output)
    {
        std::int64_t sum = biases[output];
        for (std::size_t feature = 0; feature < ABJNNUE::RuntimeLayout::InventoryContextInputs; ++feature)
            sum += std::int32_t(context[feature])
                 * std::int32_t(weights[output * ABJNNUE::RuntimeLayout::InventoryContextInputs + feature]);
        input[ABJNNUE::RuntimeLayout::AccumulatorWidth + output] =
          static_cast<std::uint8_t>(std::clamp<std::int64_t>(sum >> 7, 0, 127));
    }
    return input;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: parity package input.bin output.bin");
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        const auto model = ABJNNUE::Model::load(argv[1]);
        std::ifstream input(argv[2], std::ios::binary);
        std::ofstream output(argv[3], std::ios::binary);
        if (!input || !output) throw std::runtime_error("cannot open parity input or output");
        const auto count = read<std::uint32_t>(input);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const auto head = read<std::uint32_t>(input);
            const auto q = read<std::uint32_t>(input);
            const auto us = read<std::uint32_t>(input);
            if (head >= ABJNNUE::RuntimeLayout::LayerStacks || q > 255 || us > 1)
                throw std::runtime_error("invalid selection");

            ABJNNUE::EncodedPosition encoded;
            for (auto color : {Stockfish::WHITE, Stockfish::BLACK})
            {
                const auto n = read<std::uint32_t>(input);
                if (n > ABJNNUE::FeatureIndexList::Capacity)
                    throw std::runtime_error("too many features");
                for (std::uint32_t j = 0; j < n; ++j)
                {
                    const auto row = read<std::uint32_t>(input);
                    if (row >= ABJNNUE::RuntimeLayout::FeatureDimensions)
                        throw std::runtime_error("invalid feature row");
                    encoded.perspectives[color].active.push_back(row);
                }
            }
            ABJNNUE::InventoryContext context{};
            for (auto& value : context)
            {
                value = read<std::uint8_t>(input);
                if (value > 127) throw std::runtime_error("inventory context is outside Q0.7");
            }

            const auto accumulated = ABJNNUE::FeatureEncoder::accumulate(*model, encoded);
            const auto transformed = transform(accumulated, us);
            const auto headInput = make_head_input(*model, transformed, context);
            const auto* heads = model->eval_heads().data();
            const auto* first = heads + head * ABJNNUE::RuntimeLayout::EvalHeadBucketSize;
            const auto* second = heads + std::min(head + 1, 15U)
                                       * ABJNNUE::RuntimeLayout::EvalHeadBucketSize;
            const auto a = ABJNNUE::Layers::propagate_scalar(first, headInput.data());
            const auto b = ABJNNUE::Layers::propagate_scalar(second, headInput.data());
            const auto scalar = head == 15 ? a : static_cast<std::int32_t>(
              ABJNNUE::RuntimeLayout::interpolate_q8(a, b, static_cast<std::uint8_t>(q)));
            const auto simd = model->propagate_interpolated(head, static_cast<std::uint8_t>(q),
                                                            transformed.data(), context);

            for (const auto& perspective : accumulated.perspectives)
                output.write(reinterpret_cast<const char*>(perspective.values.data()), 2048 * 2);
            output.write(reinterpret_cast<const char*>(headInput.data()), headInput.size());
            output.write(reinterpret_cast<const char*>(&scalar), sizeof(scalar));
            output.write(reinterpret_cast<const char*>(&simd), sizeof(simd));
            if (scalar != simd) throw std::runtime_error("scalar/BMI2 mismatch");
        }
        if (!output) throw std::runtime_error("cannot write parity output");
        std::cout << "parity samples=" << count << " backend=" << ABJNNUE::Layers::backend_name() << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
