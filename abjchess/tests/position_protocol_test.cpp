#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitboard.h"
#include "engine.h"
#include "position.h"
#include "types.h"
#include "uci.h"

namespace {

using namespace Stockfish;

constexpr auto KingsOnlyFEN = "4k4/9/9/9/9/9/9/9/9/4K4 w - 0 1";
constexpr auto VisibleMoveFEN =
  "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - 0 1";
constexpr auto HiddenMoveFEN            = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w P1 0 1";
constexpr auto VisibleCapturesDarkFEN   = "4k4/9/9/x8/R3P4/9/9/9/9/4K4 w p1 0 1";
constexpr auto HiddenCapturesVisibleFEN = "4k4/9/9/9/4P4/p8/X8/9/9/4K4 w P1 0 1";
constexpr auto HiddenCapturesDarkFEN    = "4k4/9/1x7/9/1P2P4/9/9/1X7/9/4K4 w C1c1 0 1";

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void expect_position_rejected_preserving(const std::string& fen, const std::string& label) {
    std::deque<StateInfo> states(2);
    Position              position;
    position.set(KingsOnlyFEN, &states[0]);
    const std::string before = position.fen();

    bool rejected = false;
    try
    { position.set(fen, &states[1]); } catch (const std::invalid_argument&)
    { rejected = true; }

    require(rejected, label + ": malformed FEN was accepted");
    require(position.fen() == before, label + ": rejected FEN changed the active position");
}

void test_strict_five_field_fen() {
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 x - 0 1", "invalid side");
    expect_position_rejected_preserving("4k3/9/9/9/9/9/9/9/9/4K4 w - 0 1", "short rank");
    expect_position_rejected_preserving("4k4R/9/9/9/9/9/9/9/9/4K4 w - 0 1", "wide rank");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/4K4 w - 0 1", "missing rank");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/1X7/9/9/4K4 w P1 0 1",
                                        "dark piece off its covered square");
    expect_position_rejected_preserving("4k4/9/9/9/4P4/9/X8/9/9/4K4 w - 0 1",
                                        "dark piece without an identity candidate");
    expect_position_rejected_preserving("4k4/9/9/9/4P4/9/X8/9/9/4K4 w  0 1",
                                        "dark piece with an empty identity pool");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/3?K4 w - 0 1",
                                        "invalid board character");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w K1 0 1",
                                        "king in identity pool");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w R6 0 1",
                                        "identity count above inventory");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w R1R1 0 1",
                                        "duplicate identity entry");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w - -1 1",
                                        "negative halfmove clock");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w - -0 1",
                                        "signed zero halfmove clock");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w - 0 -1",
                                        "negative fullmove clock");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w - 999999999999999999999 1",
                                        "overflowing halfmove clock");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w - 0 1 extra", "extra field");
    expect_position_rejected_preserving("4k4/9/9/9/9/9/9/9/9/4K4 w - 0",
                                        "missing fullmove clock");
}

void test_empty_rest_piece_field() {
    const std::string board = "4k4/9/9/9/9/9/9/9/9/4K4 ";
    Engine            engine;
    for (const std::string side : {"w", "b"})
    {
        const std::string canonical = board + side + " - 1 8";
        for (const std::string tail : {"  1 8", "\t\t1\t8", " 1 8", " - 1 8", " - - 1 8"})
        {
            Position  position;
            StateInfo state;
            position.set(board + side + tail, &state);
            require(position.fen() == canonical,
                    "empty rest-piece field changed the position or move counters");
            engine.set_position(board + side + tail, {});
            require(engine.fen() == canonical,
                    "engine did not accept the empty rest-piece field");
        }
    }
}

void expect_engine_rejected_preserving(Engine&                         engine,
                                       const std::string&              fen,
                                       const std::vector<std::string>& moves,
                                       const std::string&              label) {
    const std::string before   = engine.fen();
    bool              rejected = false;
    try
    { engine.set_position(fen, moves); } catch (const std::invalid_argument&)
    { rejected = true; }
    require(rejected, label + ": malformed move history was accepted");
    require(engine.fen() == before, label + ": rejected move history changed the active position");
}

void test_move_token_validation_and_transactions() {
    Engine engine;
    engine.set_position(VisibleMoveFEN, {});

    Position  visible;
    StateInfo visibleState;
    visible.set(VisibleMoveFEN, &visibleState);
    require(UCIEngine::to_move(visible, "a3a4R") == Move::none(),
            "to_move accepted characters after a coordinate move");

    expect_engine_rejected_preserving(engine, VisibleMoveFEN, {"a3a4R"},
                                      "suffix on a fully visible move");
    expect_engine_rejected_preserving(engine, HiddenMoveFEN, {"a3a4"},
                                      "missing moved-dark identity");
    expect_engine_rejected_preserving(engine, HiddenMoveFEN, {"a3a4p"},
                                      "wrong-color moved-dark identity");
    expect_engine_rejected_preserving(engine, HiddenMoveFEN, {"a3a4Z"},
                                      "invalid identity character");
    expect_engine_rejected_preserving(engine, HiddenMoveFEN, {"a3a4R"},
                                      "identity absent from pool");
    expect_engine_rejected_preserving(engine, HiddenMoveFEN, {"a3a4P", "z9z8"},
                                      "invalid move after a valid prefix");
    expect_engine_rejected_preserving(engine, VisibleCapturesDarkFEN, {"a5a6P"},
                                      "wrong-color captured-dark identity");
    expect_engine_rejected_preserving(engine, HiddenCapturesDarkFEN, {"b2b7CcZ"},
                                      "overlong hidden-capture suffix");

    engine.set_position(HiddenMoveFEN, {"a3a4P"});
    require(engine.fen() == "4k4/9/9/9/4P4/P8/9/9/9/4K4 b - 1 1",
            "valid moved-dark identity was not revealed and removed");

    engine.set_position(VisibleCapturesDarkFEN, {"a5a6"});
    require(engine.fen() == "4k4/9/9/R8/4P4/9/9/9/9/4K4 b p1 0 1",
            "unknown captured-dark identity incorrectly changed the pool");
    engine.set_position(VisibleCapturesDarkFEN, {"a5a6p"});
    require(engine.fen() == "4k4/9/9/R8/4P4/9/9/9/9/4K4 b - 0 1",
            "known captured-dark identity was not removed");

    engine.set_position(HiddenCapturesVisibleFEN, {"a3a4P"});
    require(engine.fen() == "4k4/9/9/9/4P4/P8/9/9/9/4K4 b - 0 1",
            "dark mover capture did not reveal its identity");

    engine.set_position(HiddenCapturesDarkFEN, {"b2b7C"});
    require(engine.fen() == "4k4/9/1C7/9/1P2P4/9/9/9/9/4K4 b c1 0 1",
            "unknown dark victim incorrectly changed the opponent pool");
    engine.set_position(HiddenCapturesDarkFEN, {"b2b7Cc"});
    require(engine.fen() == "4k4/9/1C7/9/1P2P4/9/9/9/9/4K4 b - 0 1",
            "known dark victim identity was not removed");
}

}  // namespace

int main() {
    try
    {
        Bitboards::init();
        Position::init();
        test_strict_five_field_fen();
        test_empty_rest_piece_field();
        test_move_token_validation_and_transactions();
        std::cout << "position protocol tests passed\n";
        return 0;
    } catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
