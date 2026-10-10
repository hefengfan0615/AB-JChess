#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "bitboard.h"
#include "engine.h"
#include "misc.h"
#include "position.h"
#include "reveal.h"
#include "search.h"
#include "uci.h"

namespace {

using namespace Stockfish;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

bool is_valid_utf8(const std::string& text) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t index = 0;
    while (index < text.size())
    {
        const unsigned char first = bytes[index++];
        if (first <= 0x7F)
            continue;
        int trailing = 0;
        std::uint32_t codepoint = 0;
        if (first >= 0xC2 && first <= 0xDF)
        {
            trailing = 1;
            codepoint = first & 0x1F;
        }
        else if (first >= 0xE0 && first <= 0xEF)
        {
            trailing = 2;
            codepoint = first & 0x0F;
        }
        else if (first >= 0xF0 && first <= 0xF4)
        {
            trailing = 3;
            codepoint = first & 0x07;
        }
        else
            return false;
        if (index + trailing > text.size())
            return false;
        for (int offset = 0; offset < trailing; ++offset)
        {
            const unsigned char next = bytes[index++];
            if ((next & 0xC0) != 0x80)
                return false;
            codepoint = (codepoint << 6) | (next & 0x3F);
        }
        if ((trailing == 1 && codepoint < 0x80) || (trailing == 2 && codepoint < 0x800)
            || (trailing == 3 && codepoint < 0x10000) || codepoint > 0x10FFFF
            || (codepoint >= 0xD800 && codepoint <= 0xDFFF))
            return false;
    }
    return true;
}

void test_engine_identity() {
    require(is_valid_utf8(engine_info(false)), "startup banner is not valid UTF-8");
    const std::string uciInfo = engine_info(true);
    require(is_valid_utf8(uciInfo), "UCI identity is not valid UTF-8");
    require(uciInfo.find("0.2d") != std::string::npos, "engine identity is not 0.2d");
}

void test_rounding_and_search_contracts() {
    require(Reveal::round_divide(1, 2) == 1, "positive half did not round away from zero");
    require(Reveal::round_divide(-1, 2) == -1, "negative half did not round away from zero");
    require(Reveal::round_divide(1, 3) == 0, "positive fraction rounded up");
    require(Reveal::round_divide(-1, 3) == 0, "negative fraction rounded down");
    require(Search::chance_branch_depth(Depth(7), 0) == 7,
            "first active flip identity lost full depth");
    require(Search::chance_branch_depth(Depth(7), 1) == 6,
            "later active flip identity did not lose one ply");
    require(Search::probcut_context(Value(500), Depth(9), true).beta == 564,
            "dark-capture ProbCut beta changed");
}

void test_quiet_features_and_bonus() {
    using Reveal::QuietFeatureInputs;
    using Reveal::QuietSafety;
    const auto features = Reveal::make_quiet_features(
      QuietFeatureInputs{{2, 0, 2, 0, 2, 0}, QuietSafety::Safe});
    require(features.safety == 1024, "quiet safety feature changed");
    require(features.highValue == 1024, "high-value pool feature changed");

    Reveal::Parameters parameters;
    require(!Reveal::quiet_bonus_enabled(parameters), "zero quiet weights enabled bonus");
    require(Reveal::quiet_bonus(parameters, features) == 0, "disabled quiet bonus was nonzero");
    parameters.quietBase = 7;
    require(Reveal::quiet_bonus_enabled(parameters), "quiet base did not enable bonus");
    require(Reveal::quiet_bonus(parameters, {0, 0}) == 7,
            "quiet base bonus changed");
}

void test_reward_window() {
    require(Reveal::apply_bonus(Value(25), 15) == Value(40), "bonus application changed");
    require(Reveal::apply_bonus(VALUE_MATE_IN_MAX_PLY, 15) == VALUE_MATE_IN_MAX_PLY,
            "mate score was adjusted");
    const auto window = Reveal::child_window(Value(-20), Value(40), 15);
    require(window.alpha == Value(-25) && window.beta == Value(35),
            "bonus child window changed");

}

void test_quiet_safety() {
    constexpr auto Safe = "4k4/9/9/9/4P4/9/X8/9/9/R3K4 w P1 0 1";
    constexpr auto Contested = "4k4/9/9/9/4P4/8r/X8/9/9/R3K4 w P1 0 1";
    constexpr auto Loose = "4k4/9/9/9/4P4/8r/X8/9/9/4K4 w P1 0 1";
    const std::array fixtures = std::array{
      std::pair{Safe, Reveal::QuietSafety::Safe},
      std::pair{Contested, Reveal::QuietSafety::Contested},
      std::pair{Loose, Reveal::QuietSafety::Loose}};
    for (const auto& fixture : fixtures)
    {
        StateInfo state;
        Position position;
        position.set(fixture.first, &state);
        const Move move = UCIEngine::to_move(position, "a3a4");
        require(move.is_ok() && position.move_dark(move) && !position.capture(move)
                  && position.legal(move),
                "quiet safety fixture move is invalid");
        require(Reveal::quiet_safety(position, move) == fixture.second,
                "quiet safety classification changed");
    }
}

void set_option(Engine& engine, const std::string& name, int value) {
    std::istringstream command("name " + name + " value " + std::to_string(value));
    engine.get_options().setoption(command);
}

void test_uci_options_and_snapshot(Engine& engine) {
    auto& options = engine.get_options();
    std::ostringstream listing;
    listing << options;
    const std::string text = listing.str();

    constexpr std::array<const char*, 8> Removed = {
      "RevealBonusBase", "RevealBonusPhase", "RevealBonusPool", "RevealBonusUnknown",
      "RevealBonusComeback", "RevealMoveOrder", "RevealReduction", "RevealPruningMargin"};
    for (const char* name : Removed)
        require(options.count(name) == 0
                  && text.find("option name " + std::string(name)) == std::string::npos,
                std::string("removed V1 option remains: ") + name);

    constexpr std::array<const char*, 5> Inactive = {
      "RevealQuietPhase", "RevealQuietDiversity", "RevealQuietComeback",
      "RevealQuietMoveOrder", "RevealQuietReduction"};
    for (const char* name : Inactive)
        require(options.count(name) == 0
                  && text.find("option name " + std::string(name)) == std::string::npos,
                std::string("inactive quiet option remains: ") + name);

    struct Spec {
        const char* name;
        int         minimum;
        int         maximum;
        int         defaultValue;
    };
    constexpr Spec specs[] = {
      {"RevealQuietBase", -96, 96, -12},
      {"RevealQuietSafety", -192, 192, 24},
      {"RevealQuietHighValue", -96, 96, -12}};
    for (const Spec& spec : specs)
    {
        require(options.count(spec.name) == 1, std::string(spec.name) + " is missing");
        require(int(options[spec.name]) == spec.defaultValue,
                std::string(spec.name) + " default changed");
        set_option(engine, spec.name, spec.minimum);
        require(int(options[spec.name]) == spec.minimum,
                std::string(spec.name) + " rejected minimum");
        set_option(engine, spec.name, spec.maximum);
        require(int(options[spec.name]) == spec.maximum,
                std::string(spec.name) + " rejected maximum");
        set_option(engine, spec.name, spec.defaultValue);
    }

    set_option(engine, "RevealQuietBase", 11);
    set_option(engine, "RevealQuietSafety", 12);
    set_option(engine, "RevealQuietHighValue", 13);
    const Reveal::Parameters parameters = Search::snapshot_reveal_parameters(options);
    require(parameters.quietBase == 11 && parameters.quietSafety == 12
              && parameters.quietHighValue == 13,
            "quiet parameter snapshot changed");
    for (const Spec& spec : specs)
        set_option(engine, spec.name, spec.defaultValue);
}

void test_search_path(Engine& engine) {
    constexpr auto Fen = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w P1 0 1";
    bool           gotBaseline = false;
    int            baselineScore = 0;
    std::size_t    baselineNodes = 0;
    std::string    bestMove;
    engine.set_on_update_no_moves([](const Engine::InfoShort&) {});
    engine.set_on_update_full([&](const Engine::InfoFull& info) {
        if (info.depth == 4)
        {
            gotBaseline   = true;
            baselineScore = info.score.get<Score::InternalUnits>().value;
            baselineNodes = info.nodes;
        }
    });
    engine.set_on_bestmove([&](std::string_view move, std::string_view) {
        bestMove.assign(move);
    });
    engine.set_position(Fen, {});
    const std::string before = engine.fen();
    Search::LimitsType limits;
    limits.depth = 4;
    engine.go(limits);
    engine.wait_for_search_finished();
    require(engine.fen() == before, "search changed the root position");
    require(gotBaseline, "depth-4 baseline was not reported");
    require(bestMove == "a3a4", "default best move differs from the 0.2c baseline");
    require(baselineScore == 3, "default score differs from the 0.2c baseline");
    require(baselineNodes == 33, "default node count differs from the 0.2c baseline");
}

}  // namespace

int main() {
    try
    {
        Bitboards::init();
        Position::init();
        Engine engine;
        const char* package = std::getenv("ABJCHESS_V11_TEST_PACKAGE");
        engine.load_big_network(package ? package : "abjchess-v11.nnue");
        engine.verify_networks();
        test_engine_identity();
        test_rounding_and_search_contracts();
        test_quiet_features_and_bonus();
        test_reward_window();
        test_quiet_safety();
        test_uci_options_and_snapshot(engine);
        test_search_path(engine);
        std::cout << "reveal tests passed\n";
        return 0;
    } catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
