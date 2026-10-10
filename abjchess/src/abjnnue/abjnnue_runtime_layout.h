#ifndef ABJNNUE_RUNTIME_LAYOUT_H_INCLUDED
#define ABJNNUE_RUNTIME_LAYOUT_H_INCLUDED

#include "abjnnue_package.h"

#include <array>

namespace ABJNNUE {

struct RuntimeLayout {
    static constexpr std::size_t AccumulatorWidth         = 2048;
    static constexpr std::size_t PerspectiveOutputWidth   = AccumulatorWidth / 2;
    static constexpr std::size_t PieceSquarePlanes        = 15;
    static constexpr std::size_t Squares                 = 90;
    static constexpr std::size_t BasePieceSquareDimensions = PieceSquarePlanes * Squares;
    static constexpr std::size_t MetaDimensions           = 100;
    static constexpr std::size_t PieceSquareDimensions    = BasePieceSquareDimensions + MetaDimensions;
    static constexpr std::size_t KingBuckets              = 6;
    static constexpr std::size_t AttackBuckets            = 4;
    static constexpr std::size_t FeatureBuckets           = KingBuckets * AttackBuckets;
    static constexpr std::size_t BoardFeatureDimensions   = BasePieceSquareDimensions * FeatureBuckets;
    static constexpr std::size_t DarkRestDimensions       = MetaDimensions;
    static constexpr std::size_t VariantFeatureDimensions = PieceSquareDimensions * FeatureBuckets;
    static constexpr std::size_t FeatureDimensions        = VariantFeatureDimensions;
    static constexpr std::size_t LayerStacks              = 16;
    static constexpr std::size_t InventoryContextInputs   = 16;
    static constexpr std::size_t InventoryContextHidden   = 16;
    static constexpr std::size_t HeadInputWidth           = AccumulatorWidth + InventoryContextHidden;
    static constexpr std::int64_t InterpolationQBits       = 8;
    static constexpr std::int64_t InterpolationDenominator = (1LL << InterpolationQBits) - 1LL;

    static constexpr std::int64_t interpolate_q8(std::int64_t first,
                                                  std::int64_t second,
                                                  std::uint8_t blendQ8) noexcept {
        if (blendQ8 == 0) return first;
        if (blendQ8 == InterpolationDenominator) return second;
        const std::int64_t q = blendQ8;
        return (first * (InterpolationDenominator - q) + second * q
                + InterpolationDenominator / 2)
             / InterpolationDenominator;
    }

    // The primary chunk contains the transformer and context weights.
    // every exported weight package self-describing through its chunk sizes.
    static constexpr std::size_t TransformerBiasesSize = AccumulatorWidth * sizeof(std::int16_t);
    static constexpr std::size_t FeatureWeightsSize =
      FeatureDimensions * AccumulatorWidth * sizeof(std::int16_t);
    static constexpr std::size_t TransformerPayloadSize = TransformerBiasesSize + FeatureWeightsSize;
    static constexpr std::size_t ContextBiasesSize = InventoryContextHidden * sizeof(std::int32_t);
    static constexpr std::size_t ContextWeightsSize =
      InventoryContextInputs * InventoryContextHidden * sizeof(std::int8_t);
    static constexpr std::size_t PrimarySize = TransformerPayloadSize + ContextBiasesSize + ContextWeightsSize;
    static constexpr std::size_t L1BiasOffset = 0;
    static constexpr std::size_t L1WeightOffset = 128;
    static constexpr std::size_t L2BiasOffset = L1WeightOffset + HeadInputWidth * 32;
    static constexpr std::size_t L2WeightOffset = L2BiasOffset + 128;
    static constexpr std::size_t FinalBiasOffset = L2WeightOffset + 64 * 32;
    static constexpr std::size_t FinalWeightOffset = FinalBiasOffset + 64;
    static constexpr std::size_t EvalHeadBucketSize = FinalWeightOffset + 128;
    static constexpr std::size_t EvalHeadsSize = LayerStacks * EvalHeadBucketSize;
    static constexpr std::size_t ProbabilityScoreToMassSize = 4001;
    // Probability mass now covers the symmetric inclusive range [-950, 950].
    // Keep the inverse table length in lockstep with the ABI range.
    static constexpr std::size_t ProbabilityMassToScoreSize = 1901;

    static_assert(FeatureDimensions == 34800);
    static_assert(TransformerPayloadSize == 142544896);
    static_assert(PrimarySize == 142545216);
    static_assert(HeadInputWidth == 2064);
    static_assert(L1WeightOffset % 64 == 0 && L2BiasOffset % 64 == 0);
    static_assert(L2WeightOffset % 64 == 0 && FinalBiasOffset % 64 == 0);
    static_assert(FinalWeightOffset % 64 == 0);
    static_assert(L2BiasOffset == 0x10280);
    static_assert(L2WeightOffset == 0x10300);
    static_assert(FinalBiasOffset == 0x10B00);
    static_assert(FinalWeightOffset == 0x10B40);
    static_assert(EvalHeadBucketSize == 68544);

    ByteView transformerBiases;
    ByteView featureWeights;
    ByteView contextBiases;
    ByteView contextWeights;
    ByteView evalHeads;
    ByteView probabilityScoreToMass;
    ByteView probabilityMassToScore;

    static RuntimeLayout bind(const Package& package);
    static RuntimeLayout bind(Package&&) = delete;
    static RuntimeLayout bind(const Package&&) = delete;
    bool inference_complete() const noexcept;
    bool probability_complete() const noexcept;
};

using InventoryContext = std::array<std::uint8_t, RuntimeLayout::InventoryContextInputs>;

}  // namespace ABJNNUE

#endif
