/*
  AB-JChess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef REVEAL_H_INCLUDED
#define REVEAL_H_INCLUDED

#include <array>
#include <cstdint>
#include <utility>

#include "types.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Reveal {

struct Parameters {
    int quietBase      = 0;
    int quietSafety    = 0;
    int quietHighValue = 0;
};

enum class QuietSafety : std::int8_t { Safe, Contested, Loose };

struct QuietMoveContext {
    bool        eligible = false;
    QuietSafety safety   = QuietSafety::Contested;

    bool safe() const { return eligible && safety == QuietSafety::Safe; }
};

struct QuietFeatureInputs {
    std::array<int, 6> poolCounts;
    QuietSafety        safety;
};

struct QuietFeatures {
    int safety;
    int highValue;
};

struct Window {
    Value alpha;
    Value beta;
};

std::int64_t round_divide(std::int64_t numerator, std::int64_t denominator);
QuietFeatures make_quiet_features(const QuietFeatureInputs& inputs);
QuietFeatures make_quiet_features(const Position& position,
                                  QuietSafety    safety);
QuietSafety  quiet_safety(const Position& position, Move move);
QuietMoveContext quiet_move_context(const Position& position, Move move, bool enabled);
bool         qsearch_quiet_context_enabled(bool useReveal, bool inCheck, bool safetyNeeded);
bool         quiet_bonus_enabled(const Parameters& parameters);
int          quiet_bonus(const Parameters& parameters, const QuietFeatures& features);
Value        apply_bonus(Value rawValue, int revealBonus);
Value        first_raw(Value adjustedThreshold, int revealBonus);
Window       child_window(Value alpha, Value beta, int revealBonus);

template<typename ChildSearch>
Value search_with_bonus(Value alpha, Value beta, int revealBonus, ChildSearch&& childSearch) {
    const Window window = child_window(alpha, beta, revealBonus);
    const Value  child  = std::forward<ChildSearch>(childSearch)(window.alpha, window.beta);
    return apply_bonus(Value(-child), revealBonus);
}

}  // namespace Stockfish::Reveal

#endif  // #ifndef REVEAL_H_INCLUDED
