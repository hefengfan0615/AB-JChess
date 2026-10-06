#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "bitboard.h"
#include "position.h"
#include "types.h"
#include "uci.h"

namespace {

using namespace Stockfish;

constexpr auto KingsOnlyFEN = "4k4/9/9/9/9/9/9/9/9/4K4 w - 0 1";
constexpr auto HiddenMoveFEN = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w P1 0 1";
constexpr auto VisibleCapturesDarkFEN = "4k4/9/9/x8/R3P4/9/9/9/9/4K4 w p1 0 1";

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

Move parse_move(Position& position, const char* uci) {
    const Move move = UCIEngine::to_move(position, uci);
    require(move.is_ok(), std::string("invalid fixture move: ") + uci + " in " + position.fen());
    require(position.legal(move), std::string("illegal fixture move: ") + uci);
    return move;
}

void play(Position& position, std::deque<StateInfo>& states, const char* uci) {
    const Move move = parse_move(position, uci);
    states.emplace_back();
    position.do_move(move, states.back(), position.gives_check(move), nullptr);
}

void require_empty_skyrule_state(const StateInfo& state, const std::string& context) {
    require(state.skyruleRawChased == Bitboard(0), context + ": raw chase was not cleared");
    require(state.skyruleChased == Bitboard(0), context + ": chase was not cleared");
    require(state.skyruleAction == NONE, context + ": action was not reset to NONE");
    require(state.skyruleCount == 0, context + ": count was not reset");
}

void test_action_schema_and_empty_root() {
    static_assert(std::is_same_v<decltype(StateInfo::skyruleRawChased), Bitboard>);
    static_assert(std::is_same_v<decltype(StateInfo::skyruleChased), Bitboard>);
    static_assert(std::is_same_v<decltype(StateInfo::skyruleAction), SkyruleAction>);
    static_assert(std::is_same_v<decltype(StateInfo::skyruleCount), int>);
    static_assert(NONE == 0 && CHECK == 1 && CHASE == 2,
                  "SkyRule action values are part of the engine state contract");

    std::deque<StateInfo> states(1);
    Position              position;
    position.set(KingsOnlyFEN, &states.front());
    require_empty_skyrule_state(*position.state(), "root state");
}

void test_check_action_and_first_count() {
    // The rook move e2-d2 is intentionally a cross-file check: the pre-move
    // position is quiet, while e2-d2 opens the d-file onto the black king.
    constexpr auto CheckFEN = "r2k5/9/9/9/9/9/9/4R4/9/3K5 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(CheckFEN, &states.front());

    const Move move = parse_move(position, "e2d2");
    require(position.gives_check(move), "check fixture move does not give check");
    states.emplace_back();
    position.do_move(move, states.back(), true, nullptr);

    require(position.state()->skyruleAction == CHECK,
            "a checking move was not recorded as CHECK");
    require(position.state()->skyruleCount == 1,
            "the first continuous check did not start at count one");
    require(position.state()->skyruleRawChased == Bitboard(0),
            "a checking move retained chase targets");
}

void test_dark_move_interrupts_history() {
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(HiddenMoveFEN, &states.front());

    // Seed the root with a non-empty suffix. A hidden move must terminate it,
    // even if the moved shell happens to preserve a geometric attack.
    states.front().skyruleRawChased = square_bb(SQ_A4);
    states.front().skyruleChased    = square_bb(SQ_A4);
    states.front().skyruleAction    = CHASE;
    states.front().skyruleCount     = 17;

    const Move move = parse_move(position, "a3a4");
    require(position.move_dark(move), "hidden fixture move was not marked dark");
    states.emplace_back();
    position.do_move(move, states.back(), false, nullptr);

    require(position.state()->movedDark, "hidden fixture did not record a dark move");
    require_empty_skyrule_state(*position.state(), "dark move");
}

void test_capture_interrupts_history() {
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(VisibleCapturesDarkFEN, &states.front());
    states.front().skyruleRawChased = square_bb(SQ_A6);
    states.front().skyruleChased    = square_bb(SQ_A6);
    states.front().skyruleAction    = CHASE;
    states.front().skyruleCount     = 17;

    const Move move = parse_move(position, "a5a6");
    require(position.capture(move), "capture fixture move is not a capture");
    states.emplace_back();
    position.do_move(move, states.back(), position.gives_check(move), nullptr);

    require(position.state()->capturedPiece != NO_PIECE,
            "capture fixture did not record the captured shell");
    require_empty_skyrule_state(*position.state(), "capture");
}

void test_visible_capture_dark_undo_restores_visibility() {
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(VisibleCapturesDarkFEN, &states.front());
    const Bitboard originalDark = position.pieces(DARK);

    const Move move = parse_move(position, "a5a6");
    states.emplace_back();
    position.do_move(move, states.back(), position.gives_check(move), nullptr);
    require(position.state()->captureDark,
            "visible capture fixture did not record a dark capture");
    position.undo_move(move);

    require(position.pieces(DARK) == originalDark,
            "undo of a visible capture changed the hidden-piece mask");
    require(position.is_dark(SQ_A6) && !position.is_dark(SQ_A5),
            "undo of a visible capture restored visibility on the wrong square");
}

void test_dark_and_capture_boundaries_reset_next_action() {
    // Two white knights can independently create the same chase. The first
    // action is followed by a black dark move; the second action must start a
    // fresh suffix rather than inheriting the first count.
    {
        constexpr auto DarkBoundaryFEN =
          "3k5/9/9/x8/9/4r4/9/N8/1N7/5K3 w p1 0 1";
        std::deque<StateInfo> states(1);
        Position              position;
        position.set(DarkBoundaryFEN, &states.front());
        play(position, states, "b1d2");
        require(position.state()->skyruleAction == CHASE
                    && position.state()->skyruleCount == 1,
                "dark-boundary fixture did not create its first chase");
        play(position, states, "a6a5");
        play(position, states, "a2c3");
        require(position.state()->skyruleAction == CHASE
                    && position.state()->skyruleCount == 1,
                "dark move allowed the next chase to inherit the old count");
    }

    // The same check applies when the intervening move captures a hidden
    // piece. The capture is an identity boundary even though it is made by a
    // visible rook and does not itself expose a new SkyRule action.
    {
        constexpr auto CaptureBoundaryFEN =
          "3k5/9/9/9/9/r3r4/X8/N8/1N7/5K3 w P1 0 1";
        std::deque<StateInfo> states(1);
        Position              position;
        position.set(CaptureBoundaryFEN, &states.front());
        play(position, states, "b1d2");
        require(position.state()->skyruleAction == CHASE
                    && position.state()->skyruleCount == 1,
                "capture-boundary fixture did not create its first chase");
        play(position, states, "a4a3");
        play(position, states, "a2c3");
        require(position.state()->skyruleAction == CHASE
                    && position.state()->skyruleCount == 1,
                "capture boundary allowed the next chase to inherit the old count");
    }
}

void test_chase_limit_eighteen_and_nineteen() {
    // A knight move creates a new attack on a visible opposing rook. The two
    // quiet setup plies provide an own-side predecessor at state()->previous.
    constexpr auto ChaseFEN = "3k5/9/9/9/9/4r4/9/9/1N7/5K3 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(ChaseFEN, &states.front());
    play(position, states, "f0f1");
    play(position, states, "d9e9");

    const Move candidate = parse_move(position, "b1d2");
    require(position.state()->previous == &states[1],
            "fixture did not preserve the same-side predecessor");
    states[1].skyruleAction    = CHASE;
    states[1].skyruleRawChased = square_bb(SQ_E4);
    states[1].skyruleChased    = square_bb(SQ_E4);

    states[1].skyruleCount = 17;
    require(!position.forbidden_by_skyrule_jieqi(candidate),
            "the eighteenth continuous chase was incorrectly forbidden");

    states[1].skyruleCount = 18;
    require(position.forbidden_by_skyrule_jieqi(candidate),
            "the nineteenth continuous chase was not forbidden");
}

void test_real_chase_snapshot_and_undo() {
    constexpr auto ChaseFEN = "3k5/9/9/9/9/4r4/9/9/1N7/5K3 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(ChaseFEN, &states.front());

    play(position, states, "f0f1");
    play(position, states, "d9e9");
    const Move chase = parse_move(position, "b1d2");
    require(!position.gives_check(chase), "real chase fixture unexpectedly gives check");

    states.emplace_back();
    position.do_move(chase, states.back(), false, nullptr);
    require(position.state()->skyruleAction == CHASE,
            "a legal newly-created attack was not classified as CHASE");
    require(position.state()->skyruleRawChased & square_bb(SQ_E4),
            "the real chase target was not present in the raw delta");
    require(position.state()->skyruleCount == 1,
            "a real chase did not start a fresh continuous suffix");

    position.undo_move(chase);
    require(position.state() == &states[2], "undo did not restore the chase predecessor state");
    require(position.piece_on(SQ_B1) == W_KNIGHT && position.piece_on(SQ_E4) == B_ROOK,
            "undo did not restore stable chase piece locations");
}

void test_true_and_pinned_roots() {
    constexpr auto TrueRootFEN = "2k6/9/9/9/2n6/4c4/9/9/1N7/5K3 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(TrueRootFEN, &states.front());

    const Move move = parse_move(position, "b1d2");
    states.emplace_back();
    position.do_move(move, states.back(), position.gives_check(move), nullptr);

    require(position.state()->skyruleAction == NONE,
            "a legal true root did not neutralize the knight attack");
    Bitboard rooted = 0;
    position.skyrule_chased(&rooted);
    require(rooted & square_bb(SQ_E4), "true-root target was not recorded as rooted");

    std::deque<StateInfo> pinnedStates(1);
    Position              pinned;
    pinned.set("2k6/9/9/9/2n6/4c4/9/9/1N7/2R2K3 w - 0 1", &pinnedStates.front());
    const Move pinnedMove = parse_move(pinned, "b1d2");
    pinnedStates.emplace_back();
    pinned.do_move(pinnedMove, pinnedStates.back(), pinned.gives_check(pinnedMove), nullptr);
    require(pinned.state()->skyruleAction == CHASE,
            "a pinned root incorrectly neutralized the knight attack");
    require(pinned.state()->skyruleRawChased & square_bb(SQ_E4),
            "pinned-root chase target was not recorded");
}

void test_same_type_and_cross_region_roots() {
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set("3k5/9/9/9/9/4n4/9/9/1N7/5K3 w - 0 1", &states.front());
        play(position, states, "b1d2");
        require(position.state()->skyruleAction == NONE,
                "same-type horse recapture was not a neutralizer");
    }
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set("3k5/9/9/9/9/4n4/4p4/9/1N7/5K3 w - 0 1", &states.front());
        play(position, states, "b1d2");
        require(position.state()->skyruleAction == CHASE,
                "blocked same-type horse recapture incorrectly neutralized a chase");
    }
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set("3k5/9/9/6b2/9/4c4/9/9/1N7/5K3 w - 0 1", &states.front());
        play(position, states, "b1d2");
        Bitboard rooted = 0;
        position.skyrule_chased(&rooted);
        require(position.state()->skyruleAction == NONE && (rooted & square_bb(SQ_E4)),
                "revealed cross-river elephant root was not honored");
    }
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set("3k5/9/9/6b2/5p3/4c4/9/9/1N7/5K3 w - 0 1", &states.front());
        play(position, states, "b1d2");
        require(position.state()->skyruleAction == CHASE,
                "blocked elephant eye incorrectly neutralized a chase");
    }
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set("3k5/9/9/9/5a3/4c4/9/9/1N7/5K3 w - 0 1", &states.front());
        play(position, states, "b1d2");
        Bitboard rooted = 0;
        position.skyrule_chased(&rooted);
        require(position.state()->skyruleAction == NONE && (rooted & square_bb(SQ_E4)),
                "revealed cross-river advisor root was not honored");
    }
}

void test_stable_target_relay_and_real_dark_capture_reset() {
    std::deque<StateInfo> states(1);
    Position              position;
    position.set("4k4/9/9/9/9/1c7/9/9/R8/5K3 w - 0 1", &states.front());

    play(position, states, "a1b1");
    require(position.state()->skyruleAction == CHASE,
            "stable-target fixture did not create the initial chase");
    play(position, states, "b4c4");
    play(position, states, "b1c1");
    require(position.state()->skyruleAction == CHASE
                && position.state()->skyruleCount == 2,
            "target movement did not relay the stable chase identity");

    std::deque<StateInfo> darkStates(1);
    Position              dark;
    dark.set("3k5/9/9/x8/R8/4r4/9/9/1N7/5K3 w p1 0 1", &darkStates.front());
    play(dark, darkStates, "b1d2");
    play(dark, darkStates, "d9e9");
    const Move capture = parse_move(dark, "a5a6");
    require(dark.capture(capture) && dark.is_dark(capture.to_sq()),
            "dark-capture fixture was not a capture of a hidden piece");
    darkStates.emplace_back();
    dark.do_move(capture, darkStates.back(), dark.gives_check(capture), nullptr);
    require(dark.state()->capturedPiece != NO_PIECE && dark.state()->captureDark,
            "dark capture did not record the reveal/capture transition");
    require_empty_skyrule_state(*dark.state(), "real dark capture");
}

void test_real_cycles_defer_at_root_and_adjudicate_inside() {
    {
        std::deque<StateInfo> states(1);
        Position              position;
        position.set("4k4/9/9/9/9/1c7/9/9/R8/5K3 w - 0 1", &states.front());
        play(position, states, "a1b1");
        play(position, states, "b4a4");
        play(position, states, "b1a1");
        play(position, states, "a4b4");
        Value result = VALUE_ZERO;
        require(!position.rule_judge(result, 0, CHASING_RULE_SKYRULE_JIEQI)
                    && result == VALUE_NONE,
                "real chase cycle was not deferred at the root");
        result = VALUE_ZERO;
        require(position.rule_judge(result, 4, CHASING_RULE_SKYRULE_JIEQI)
                    && result == mated_in(4),
                "real chase cycle used the wrong internal responsibility/value");
    }
    {
        std::deque<StateInfo> states(1);
        Position              position;
        position.set("3k5/4R4/9/9/9/9/9/9/9/5K3 w - 0 1", &states.front());
        play(position, states, "e8d8");
        play(position, states, "d9e9");
        play(position, states, "d8e8");
        play(position, states, "e9d9");
        Value result = VALUE_ZERO;
        require(!position.rule_judge(result, 0, CHASING_RULE_SKYRULE_JIEQI)
                    && result == VALUE_NONE,
                "real check cycle was not deferred at the root");
        result = VALUE_ZERO;
        require(position.rule_judge(result, 4, CHASING_RULE_SKYRULE_JIEQI)
                    && result == mated_in(4),
                "real check cycle used the wrong internal responsibility/value");
    }
}

void test_single_checker_limit() {
    constexpr auto CheckFEN = "r2k5/9/9/9/9/9/9/4R4/9/3K5 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(CheckFEN, &states.front());
    play(position, states, "d0e0");
    play(position, states, "a9a8");

    const Move candidate = parse_move(position, "e2d2");
    states[1].skyruleAction = CHECK;
    states[1].skyruleCount  = 5;
    require(!position.forbidden_by_skyrule_jieqi(candidate),
            "the sixth continuous check was incorrectly forbidden");

    states[1].skyruleCount = 6;
    require(position.forbidden_by_skyrule_jieqi(candidate),
            "the seventh continuous check was not forbidden");
}

void test_real_single_checker_sixth_allowed_seventh_forbidden() {
    constexpr auto CheckLoopFEN = "3k5/4R4/9/9/9/9/9/9/9/5K3 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(CheckLoopFEN, &states.front());

    for (int cycle = 0; cycle < 2; ++cycle)
    {
        play(position, states, "e8d8");
        play(position, states, "d9e9");
        play(position, states, "d8e8");
        play(position, states, "e9d9");
    }

    require(position.state()->skyruleAction == NONE,
            "real check-loop reply unexpectedly retained CHECK action");
    const Move fifth = parse_move(position, "e8d8");
    require(!position.forbidden_by_skyrule_jieqi(fifth),
            "real fifth single-check action was incorrectly forbidden");
    play(position, states, "e8d8");
    play(position, states, "d9e9");
    play(position, states, "d8e8");
    play(position, states, "e9d9");
    const Move seventh = parse_move(position, "e8d8");
    require(states[states.size() - 2].skyruleCount == 6,
            "real checker loop did not reach six checking actions before boundary");
    require(position.forbidden_by_skyrule_jieqi(seventh),
            "real seventh single-check action was not forbidden");

}

void test_persistent_checker_is_adjudicated_only_inside_search() {
    // The white rook is the stable checker identity in the synthetic history.
    // On the candidate move the rook both checks and clears the knight's leg,
    // so the suffix has two checker identities but the rook has given every
    // one of its seven checks. Counting only the union would incorrectly
    // raise the limit to twelve.
    constexpr auto CheckFEN = "r2k5/4RN3/9/9/9/9/9/R8/9/5K3 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(CheckFEN, &states.front());

    for (int ply = 0; ply < 6; ++ply)
    {
        play(position, states, ply % 2 == 0 ? "a2a3" : "a3a2");
        play(position, states, ply % 2 == 0 ? "a9b9" : "b9a9");
        StateInfo& whiteCheck = states[states.size() - 2];
        whiteCheck.skyruleAction = CHECK;
        whiteCheck.skyruleCount  = ply / 2 + 1;
        whiteCheck.checkersBB    = square_bb(SQ_E8);
    }

    // Counts are per side, so the last white action is the sixth check.
    states[states.size() - 2].skyruleCount = 6;
    const Move candidate = parse_move(position, "e8d8");
    require(position.gives_check(candidate), "persistent-check fixture does not give check");
    states.emplace_back();
    position.do_move(candidate, states.back(), true, nullptr);
    require(position.state()->skyruleCount == 7,
            "persistent-check fixture did not create a seventh checking action");
    require(popcount(position.state()->checkersBB) == 2,
            "persistent-check fixture did not create two checking identities");
    position.undo_move(candidate);
    states.pop_back();
    require(!position.forbidden_by_skyrule_jieqi(candidate),
            "root move filtering rejected a persistent seventh check before search");

    states.emplace_back();
    position.do_move(candidate, states.back(), true, nullptr);

    Value rootResult = VALUE_ZERO;
    require(!position.rule_judge(rootResult, 0, CHASING_RULE_SKYRULE_JIEQI),
            "root search adjudicated an overlong persistent check immediately");

    constexpr int InternalPly = 7;
    Value internalResult = VALUE_ZERO;
    require(position.rule_judge(internalResult, InternalPly,
                                CHASING_RULE_SKYRULE_JIEQI),
            "internal search did not adjudicate an overlong persistent check");
    require(internalResult == mate_in(InternalPly - 1),
            "internal persistent-check adjudication used the wrong mate distance");
}

void test_alternating_checkers_keep_the_twelve_check_limit() {
    // Neither checker remains throughout the suffix. The normal two-identity
    // limit must therefore still allow the seventh check.
    constexpr auto CheckFEN = "r2k5/4RN3/9/9/9/9/9/R8/9/5K3 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(CheckFEN, &states.front());

    for (int ply = 0; ply < 6; ++ply)
    {
        play(position, states, ply % 2 == 0 ? "a2a3" : "a3a2");
        play(position, states, ply % 2 == 0 ? "a9b9" : "b9a9");
        StateInfo& whiteCheck = states[states.size() - 2];
        whiteCheck.skyruleAction = CHECK;
        whiteCheck.skyruleCount  = ply / 2 + 1;
        whiteCheck.checkersBB = square_bb(ply % 2 == 0 ? SQ_E8 : SQ_F8);
    }

    states[states.size() - 2].skyruleCount = 6;
    const Move candidate = parse_move(position, "e8d8");
    require(position.gives_check(candidate), "alternating-check fixture does not give check");
    require(!position.forbidden_by_skyrule_jieqi(candidate),
            "alternating checkers did not retain their twelve-check limit");
}

void test_root_repetition_is_deferred() {
    // This four-ply reversible loop reaches the old five-fold fast path. At
    // the root SkyRule detector must defer (VALUE_NONE) or leave the position
    // unjudged so search can try a breaker; VALUE_DRAW is not acceptable.
    constexpr auto LoopFEN = "r2k5/9/9/9/9/9/9/R8/9/4K4 w - 0 1";
    std::deque<StateInfo> states(1);
    Position              position;
    position.set(LoopFEN, &states.front());
    for (int cycle = 0; cycle < 5; ++cycle)
    {
        play(position, states, "a2a3");
        play(position, states, "a9a8");
        play(position, states, "a3a2");
        play(position, states, "a8a9");
    }

    // Keep the board loop quiet, but seed both sides' action snapshots as a
    // SkyRule chase. This isolates root deferral from ordinary repetition
    // adjudication while preserving the real repeated position keys.
    for (std::size_t index = 1; index < states.size(); ++index)
    {
        states[index].skyruleAction    = CHASE;
        states[index].skyruleRawChased = square_bb(SQ_D9);
        states[index].skyruleChased    = square_bb(SQ_D9);
        states[index].skyruleCount     = 1;
    }

    Value result = VALUE_ZERO;
    const bool judged = position.rule_judge(result, 0, CHASING_RULE_SKYRULE_JIEQI);
    require(!judged || result == VALUE_NONE,
            "root SkyRule repetition was converted directly to a draw");
}

}  // namespace

int main() {
    try
    {
        Bitboards::init();
        Position::init();
        test_action_schema_and_empty_root();
        test_check_action_and_first_count();
        test_dark_move_interrupts_history();
        test_capture_interrupts_history();
        test_visible_capture_dark_undo_restores_visibility();
        test_dark_and_capture_boundaries_reset_next_action();
        test_chase_limit_eighteen_and_nineteen();
        test_real_chase_snapshot_and_undo();
        test_true_and_pinned_roots();
        test_same_type_and_cross_region_roots();
        test_stable_target_relay_and_real_dark_capture_reset();
        test_real_cycles_defer_at_root_and_adjudicate_inside();
        test_single_checker_limit();
        test_real_single_checker_sixth_allowed_seventh_forbidden();
        test_persistent_checker_is_adjudicated_only_inside_search();
        test_alternating_checkers_keep_the_twelve_check_limit();
        test_root_repetition_is_deferred();
        std::cout << "skyrule jieqi tests passed\n";
        return 0;
    } catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
