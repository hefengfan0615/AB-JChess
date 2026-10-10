/*
  AB-JChess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "reveal.h"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "bitboard.h"
#include "position.h"

namespace Stockfish::Reveal {

std::int64_t round_divide(std::int64_t numerator, std::int64_t denominator) {
    assert(denominator > 0);

    const std::int64_t quotient  = numerator / denominator;
    const std::int64_t remainder = numerator % denominator;
    const std::int64_t magnitude = remainder < 0 ? -remainder : remainder;

    return quotient + (magnitude * 2 >= denominator ? (numerator < 0 ? -1 : 1) : 0);
}

QuietFeatures make_quiet_features(const QuietFeatureInputs& inputs) {
    const std::int64_t poolTotal =
      std::accumulate(inputs.poolCounts.begin(), inputs.poolCounts.end(), 0LL);
    int highValue = 0;
    if (poolTotal > 0)
    {
        const std::int64_t highCount = std::int64_t(inputs.poolCounts[0])
                                     + inputs.poolCounts[2] + inputs.poolCounts[4];
        const std::int64_t centeredHighValue = 5 * highCount - 2 * poolTotal;
        highValue = int(round_divide(centeredHighValue * 1024,
                                     (centeredHighValue >= 0 ? 3 : 2) * poolTotal));

    }

    const int safety = inputs.safety == QuietSafety::Safe    ? 1024
                     : inputs.safety == QuietSafety::Loose   ? -1024
                                                             : 0;
    return {safety, highValue};
}

QuietFeatures make_quiet_features(const Position& position, QuietSafety safety) {
    Position::RestPieceList restPieces;
    const int restPieceTypes = position.rest_pieces(position.side_to_move(), restPieces);

    std::array<int, 6> poolCounts{};
    for (int i = 0; i < restPieceTypes; ++i)
    {
        const auto [piece, count] = restPieces[i];
        switch (type_of(piece))
        {
        case ROOK: poolCounts[0] += count; break;
        case ADVISOR: poolCounts[1] += count; break;
        case CANNON: poolCounts[2] += count; break;
        case PAWN: poolCounts[3] += count; break;
        case KNIGHT: poolCounts[4] += count; break;
        case BISHOP: poolCounts[5] += count; break;
        default: assert(false); break;
        }
    }

    return make_quiet_features({poolCounts, safety});
}

QuietSafety quiet_safety(const Position& pos, Move move) {
    assert(pos.move_dark(move));
    assert(!pos.capture(move));

    const Bitboard fromBB        = square_bb(move.from_sq());
    const Bitboard toBB          = square_bb(move.to_sq());
    const Bitboard occupiedAfter = (pos.pieces() ^ fromBB) | toBB;
    const Bitboard attackers =
      pos.attackers_to(move.to_sq(), occupiedAfter) & ~pos.pieces(DARK);
    const Bitboard enemy = attackers & pos.pieces(~pos.side_to_move());
    const Bitboard friendly =
      (attackers & pos.pieces(pos.side_to_move())) & ~fromBB;

    return !enemy ? QuietSafety::Safe
                  : friendly ? QuietSafety::Contested : QuietSafety::Loose;
}

QuietMoveContext quiet_move_context(const Position& pos, Move move, bool enabled) {
    if (!enabled || !pos.move_dark(move) || pos.capture(move))
        return {};

    return {true, quiet_safety(pos, move)};
}

bool qsearch_quiet_context_enabled(bool useReveal, bool inCheck, bool safetyNeeded) {
    return useReveal && inCheck && safetyNeeded;
}

bool quiet_bonus_enabled(const Parameters& parameters) {
    return parameters.quietBase || parameters.quietSafety || parameters.quietHighValue;
}

int quiet_bonus(const Parameters& parameters, const QuietFeatures& features) {
    if (!quiet_bonus_enabled(parameters))
        return 0;

    const std::int64_t weighted = std::int64_t(parameters.quietSafety) * features.safety
                                + std::int64_t(parameters.quietHighValue) * features.highValue;

    const std::int64_t result =
      std::int64_t(parameters.quietBase) + round_divide(weighted, 1024);
    return int(std::clamp<std::int64_t>(result, -192, 192));
}

Value apply_bonus(Value rawValue, int revealBonus) {
    if (is_decisive(rawValue))
        return rawValue;

    return Value(std::clamp(int(rawValue) + revealBonus, int(VALUE_MATED_IN_MAX_PLY) + 1,
                            int(VALUE_MATE_IN_MAX_PLY) - 1));
}

Value first_raw(Value adjustedThreshold, int revealBonus) {
    constexpr int Lower  = VALUE_MATED_IN_MAX_PLY;
    constexpr int Upper  = VALUE_MATE_IN_MAX_PLY;
    const int     target = adjustedThreshold;

    if (target <= Lower || target >= Upper)
        return adjustedThreshold;
    if (target == Lower + 1)
        return Value(Lower + 1);

    const int candidate = std::max(Lower + 1, target - revealBonus);
    return Value(candidate <= Upper - 1 ? candidate : Upper);
}

Window child_window(Value alpha, Value beta, int revealBonus) {
    const Value rawAlpha = Value(first_raw(Value(alpha + 1), revealBonus) - 1);
    const Value rawBeta  = first_raw(beta, revealBonus);
    return {Value(-rawBeta), Value(-rawAlpha)};
}

}  // namespace Stockfish::Reveal
