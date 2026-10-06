#include <array>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "reveal.h"
#include "bitboard.h"
#include "engine.h"
#include "misc.h"
#include "movepick.h"
#include "position.h"
#include "uci.h"
#include "ucioption.h"

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

        int           trailing = 0;
        std::uint32_t codepoint = 0;
        if (first >= 0xC2 && first <= 0xDF)
        {
            trailing  = 1;
            codepoint = first & 0x1F;
        }
        else if (first >= 0xE0 && first <= 0xEF)
        {
            trailing  = 2;
            codepoint = first & 0x0F;
        }
        else if (first >= 0xF0 && first <= 0xF4)
        {
            trailing  = 3;
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

void test_engine_banner_is_utf8() {
    require(is_valid_utf8(engine_info(false)), "engine startup banner is not valid UTF-8");
    require(is_valid_utf8(engine_info(true)), "UCI engine identity is not valid UTF-8");
}

void test_round_half_away_from_zero() {
    require(Reveal::round_divide(1, 2) == 1, "positive half did not round away from zero");
    require(Reveal::round_divide(-1, 2) == -1, "negative half did not round away from zero");
    require(Reveal::round_divide(1, 3) == 0, "positive fraction below half rounded up");
    require(Reveal::round_divide(-1, 3) == 0, "negative fraction below half rounded down");
}

void test_integrated_search_contracts() {
    require(Search::chance_branch_depth(Depth(0), 0) == 0,
            "zero-depth active flip branch changed depth");
    require(Search::chance_branch_depth(Depth(1), 3) == 1,
            "one-ply active flip branch was reduced below one ply");
    require(Search::chance_branch_depth(Depth(7), 0) == 7,
            "first active flip identity lost full depth");
    require(Search::chance_branch_depth(Depth(7), 1) == 6,
            "later active flip identity did not lose exactly one ply");

    const Search::ProbCutContext visible =
      Search::probcut_context(Value(500), Depth(9), false);
    require(visible.beta == 500 && visible.depth == 4,
            "visible-capture ProbCut context changed from depth-5");

    const Search::ProbCutContext dark =
      Search::probcut_context(Value(500), Depth(9), true);
    require(dark.beta == 564 && dark.depth == 3,
            "dark-capture ProbCut context is not beta+64/depth-6");

    const Search::ProbCutContext shallowDark =
      Search::probcut_context(Value(VALUE_INFINITE - 16), Depth(4), true);
    require(shallowDark.beta == VALUE_INFINITE - 1 && shallowDark.depth == 0,
            "dark-capture ProbCut endpoints are not clamped");

    int reduction = 545;
    {
        Search::ChanceBranchReduction branchReduction(reduction);
        reduction = 0;
        branchReduction.restore();
        require(reduction == 545, "first identity did not inherit the parent reduction");
        reduction = -123;
        branchReduction.restore();
        require(reduction == 545, "later identity inherited a sibling reduction");
    }
    require(reduction == 0, "chance search did not clear the consumed parent reduction");
}

Reveal::Features features(int          globalDark,
                          int          poolTotal,
                          int          ownBoardDark,
                          std::int64_t poolValueAbovePawn,
                          Value        rawParentEval) {
    return Reveal::make_features(
      {globalDark, poolTotal, ownBoardDark, poolValueAbovePawn, rawParentEval});
}

void test_feature_endpoints_and_pool_counts() {
    auto f = features(0, 0, 0, 0, 0);
    require(f.phase == 0 && f.pool == 0 && f.unknown == 0 && f.comeback == 0,
            "zero feature endpoint is not zero");

    f = features(30, 2, 2, 2LL * (RookValue - PawnValue), -800);
    require(f.phase == 1024, "phase endpoint is not 1024");
    require(f.pool == 1024, "pool denominator used piece kinds instead of piece count");
    require(f.unknown == 0, "known pool produced an unknown feature");
    require(f.comeback == 1024, "comeback losing endpoint is not 1024");

    f = features(15, 2, 0, RookValue - PawnValue, 800);
    require(f.phase == 512, "phase midpoint is not 512");
    require(f.pool == 512, "mixed-value pool midpoint is not 512");
    require(f.unknown == 137, "unknown feature did not round half away from zero");
    require(f.comeback == -1024, "comeback winning endpoint is not -1024");

    f = features(1, 30, 40, 0, 4000);
    require(f.phase == 34, "phase fraction was not rounded to nearest");
    require(f.unknown == 0, "negative unknown count was not clamped");
    require(f.comeback == -1024, "comeback input was not clamped to 800");

    f = features(1, 30, 0, 0, -4000);
    require(f.unknown == 1024, "unknown count was not clamped to 15");
    require(f.comeback == 1024, "negative comeback input was not clamped to -800");
}

void test_quiet_features() {
    using Reveal::QuietFeatureInputs;
    using Reveal::QuietSafety;

    auto initial = Reveal::make_quiet_features(
      QuietFeatureInputs{30, {2, 2, 2, 5, 2, 2}, QuietSafety::Safe, Value(-800)});
    require(initial.phase == 1024, "initial quiet phase is not +1024");
    require(initial.highValue == 0 && initial.diversity == 0,
            "initial identity pool is not centered");
    require(initial.safety == 1024 && initial.comeback == 1024,
            "safe comeback endpoint is wrong");

    auto middle = Reveal::make_quiet_features(
      QuietFeatureInputs{15, {2, 2, 2, 5, 2, 2}, QuietSafety::Contested, Value(800)});
    require(middle.phase == 0 && middle.safety == 0 && middle.comeback == -512,
            "middle/contested features are wrong");

    auto low = Reveal::make_quiet_features(
      QuietFeatureInputs{0, {0, 0, 0, 15, 0, 0}, QuietSafety::Loose, Value(-800)});
    require(low.phase == -1024 && low.highValue == -1024 && low.diversity == -1024,
            "low/concentrated pool endpoints are wrong");
    require(low.safety == -1024 && low.comeback == 0, "loose gate is wrong");

    auto high = Reveal::make_quiet_features(
      QuietFeatureInputs{30, {2, 0, 2, 0, 2, 0}, QuietSafety::Safe, VALUE_ZERO});
    require(high.highValue == 1024, "high-value pool endpoint is wrong");

    auto empty = Reveal::make_quiet_features(
      QuietFeatureInputs{7, {0, 0, 0, 0, 0, 0}, QuietSafety::Safe, VALUE_ZERO});
    require(empty.highValue == 0 && empty.diversity == 0, "T=0 guard failed");

    auto fractional = Reveal::make_quiet_features(
      QuietFeatureInputs{1, {1, 1, 1, 1, 1, 1}, QuietSafety::Safe, VALUE_ZERO});
    require(fractional.phase == -956, "fractional quiet phase rounding is wrong");
    require(fractional.highValue == 171 && fractional.diversity == 43,
            "fractional quiet pool scaling or rounding is wrong");

    auto contested = Reveal::make_quiet_features(
      QuietFeatureInputs{1, {0, 0, 0, 0, 0, 0}, QuietSafety::Contested, Value(1)});
    require(contested.comeback == -1, "contested comeback gate rounding is wrong");
}

Reveal::QuietSafety safety_for(const char* fen, const char* uci) {
    StateInfo st;
    Position  pos;
    pos.set(fen, &st);
    const Move move = UCIEngine::to_move(pos, uci);
    const std::string context = std::string("quiet safety fixture move ") + uci;
    require(move.is_ok(), context + " is unavailable");
    require(pos.move_dark(move), context + " is not a dark move");
    require(!pos.capture(move), context + " is not quiet");
    require(pos.legal(move), context + " is not legal");
    return Reveal::quiet_safety(pos, move);
}

void test_quiet_safety() {
    constexpr auto Safe = "4k4/9/9/9/4P4/9/X8/9/9/R3K4 w P1 0 1";
    constexpr auto Contested = "4k4/9/9/9/4P4/8r/X8/9/9/R3K4 w P1 0 1";
    constexpr auto Loose = "4k4/9/9/9/4P4/8r/X8/9/9/4K4 w P1 0 1";
    constexpr auto Pinned = "4k4/9/9/9/9/4r4/X8/9/9/3KR4 w P1 0 1";
    constexpr auto BlackContested = "r3k4/9/9/x8/8R/4p4/9/9/9/4K4 b p1 0 1";
    constexpr auto HiddenCoverDefender =
      "4k4/9/9/r8/4P4/9/X8/9/9/X3K4 w R1P1 0 1";

    require(safety_for(Safe, "a3a4") == Reveal::QuietSafety::Safe,
            "safe reveal was not classified safe");
    require(safety_for(Contested, "a3a4") == Reveal::QuietSafety::Contested,
            "contested reveal was not classified contested");
    require(safety_for(Loose, "a3a4") == Reveal::QuietSafety::Loose,
            "loose reveal was not classified loose");
    require(safety_for(Pinned, "a3a4") == Reveal::QuietSafety::Loose,
            "pinned geometric attacker was incorrectly discarded");
    require(safety_for(BlackContested, "a6a5") == Reveal::QuietSafety::Contested,
            "black contested reveal was not classified symmetrically");
    StateInfo hiddenState;
    Position  hiddenPosition;
    hiddenPosition.set(HiddenCoverDefender, &hiddenState);
    const Move hiddenMove = UCIEngine::to_move(hiddenPosition, "a3a4");
    require(hiddenMove.is_ok(), "hidden cover fixture move a3a4 is unavailable");
    require(hiddenPosition.move_dark(hiddenMove), "hidden cover fixture move is not dark");
    require(!hiddenPosition.capture(hiddenMove), "hidden cover fixture move is not quiet");
    require(hiddenPosition.legal(hiddenMove), "hidden cover fixture move is not legal");
    const Bitboard occupiedAfter =
      (hiddenPosition.pieces() ^ hiddenMove.from_sq()) | hiddenMove.to_sq();
    const Bitboard rawAttackers =
      hiddenPosition.attackers_to(hiddenMove.to_sq(), occupiedAfter);
    require(bool(rawAttackers & SQ_A6), "visible a6 rook does not attack a4 in hidden fixture");
    require(bool(rawAttackers & SQ_A0), "hidden a0 rook cover-type is not a leak candidate");
    require(bool(hiddenPosition.pieces(DARK) & SQ_A0),
            "a0 leak candidate is not marked as a dark shell");
    const auto hiddenSafety = Reveal::quiet_safety(hiddenPosition, hiddenMove);
    require(hiddenSafety == Reveal::QuietSafety::Loose,
            "hidden a0 shell leaked its rook cover-type as an a4 defender: actual "
              + std::to_string(int(hiddenSafety)));
}

void test_bonus_rounding_clamping_and_disabled_fast_path() {
    Reveal::Parameters p{};
    require(!Reveal::bonus_enabled(p), "zero bonus weights were not disabled");
    require(Reveal::bonus(p, {1024, 1024, 1024, 1024}) == 0,
            "zero bonus weights did not return zero");

    p.bonusPhase = 1;
    require(Reveal::bonus(p, {512, 0, 0, 0}) == 1,
            "positive bonus half did not round away from zero");
    p.bonusPhase = -1;
    require(Reveal::bonus(p, {512, 0, 0, 0}) == -1,
            "negative bonus half did not round away from zero");

    p                  = {};
    p.bonusBase        = 256;
    p.bonusPhase       = 256;
    p.bonusPool        = 256;
    p.bonusUnknown     = 256;
    p.bonusComeback    = 256;
    const auto maximum = Reveal::Features{1024, 1024, 1024, 1024};
    require(Reveal::bonus(p, maximum) == 512, "positive bonus was not clamped to 512");

    p.bonusBase = p.bonusPhase = p.bonusPool = p.bonusUnknown = p.bonusComeback = -256;
    require(Reveal::bonus(p, maximum) == -512, "negative bonus was not clamped to -512");
}

void test_quiet_bonus_and_combination() {
    Reveal::Parameters p{};
    require(!Reveal::quiet_bonus_enabled(p), "zero quiet weights were not disabled");
    require(Reveal::quiet_bonus(p, {1024, 1024, 0, 0, 0}) == 0,
            "zero quiet weights did not return zero");

    p.quietBase = 7;
    require(Reveal::quiet_bonus_enabled(p), "quietBase alone did not enable quiet bonus");
    require(Reveal::quiet_bonus(p, {0, 0, 0, 0, 0}) == 7,
            "quietBase single-hot result is wrong");

    p            = {};
    p.quietPhase = 29;
    require(Reveal::quiet_bonus_enabled(p), "quietPhase alone did not enable quiet bonus");
    require(Reveal::quiet_bonus(p, {455, 0, 0, 0, 0}) == 13,
            "quietPhase single-hot result is wrong");

    p             = {};
    p.quietSafety = -17;
    require(Reveal::quiet_bonus_enabled(p), "quietSafety alone did not enable quiet bonus");
    require(Reveal::quiet_bonus(p, {0, 301, 0, 0, 0}) == -5,
            "quietSafety single-hot result is wrong");

    p                = {};
    p.quietHighValue = 23;
    require(Reveal::quiet_bonus_enabled(p), "quietHighValue alone did not enable quiet bonus");
    require(Reveal::quiet_bonus(p, {0, 0, 267, 0, 0}) == 6,
            "quietHighValue single-hot result is wrong");

    p                = {};
    p.quietDiversity = -41;
    require(Reveal::quiet_bonus_enabled(p), "quietDiversity alone did not enable quiet bonus");
    require(Reveal::quiet_bonus(p, {0, 0, 0, 193, 0}) == -8,
            "quietDiversity single-hot result is wrong");

    p               = {};
    p.quietComeback = 13;
    require(Reveal::quiet_bonus_enabled(p), "quietComeback alone did not enable quiet bonus");
    require(Reveal::quiet_bonus(p, {0, 0, 0, 0, -512}) == -7,
            "quietComeback single-hot result or negative half rounding is wrong");

    p                = {};
    p.quietMoveOrder = 91;
    require(!Reveal::quiet_bonus_enabled(p), "quietMoveOrder enabled quiet bonus");
    require(Reveal::quiet_bonus(p, {1024, 1024, 1024, 1024, 1024}) == 0,
            "quietMoveOrder produced a quiet bonus");

    p                 = {};
    p.quietReduction = -73;
    require(!Reveal::quiet_bonus_enabled(p), "quietReduction enabled quiet bonus");
    require(Reveal::quiet_bonus(p, {1024, 1024, 1024, 1024, 1024}) == 0,
            "quietReduction produced a quiet bonus");

    p            = {};
    p.quietPhase = 1;
    require(Reveal::quiet_bonus(p, {512, 0, 0, 0, 0}) == 1,
            "positive quiet half did not round away from zero");
    p.quietPhase = -1;
    require(Reveal::quiet_bonus(p, {512, 0, 0, 0, 0}) == -1,
            "negative quiet half did not round away from zero");

    p              = {};
    p.quietBase    = 96;
    p.quietPhase   = 96;
    p.quietSafety  = 192;
    require(Reveal::quiet_bonus(p, {1024, 1024, 0, 0, 0}) == 192,
            "Bq positive clamp failed");
    p.quietBase   = -96;
    p.quietPhase  = -96;
    p.quietSafety = -192;
    require(Reveal::quiet_bonus(p, {1024, 1024, 0, 0, 0}) == -192,
            "Bq negative clamp failed");

    require(Reveal::combine_bonus(400, 192, true) == 512,
            "combined +512 clamp failed");
    require(Reveal::combine_bonus(-400, -192, true) == -512,
            "combined -512 clamp failed");
    require(Reveal::combine_bonus(77, -12, true) == 65,
            "ordinary eligible bonuses were not combined");
    require(Reveal::combine_bonus(77, 192, false) == 77,
            "capture received V2 bonus");
    require(Reveal::combine_bonus(700, -192, false) == 700,
            "non-eligible V1 bonus was changed or clamped");
    require(Reveal::apply_bonus(VALUE_MATE_IN_MAX_PLY, 192) == VALUE_MATE_IN_MAX_PLY,
            "decisive score changed");
}

void test_combined_bonus_and_lazy_raw_eval_gates_follow_move_context() {
    const Reveal::QuietMoveContext noQuiet{};
    const Reveal::QuietMoveContext safeQuiet{true, Reveal::QuietSafety::Safe};

    Reveal::Parameters parameters{};
    require(!Reveal::raw_eval_needed(parameters, noQuiet),
            "all-zero parameters requested a raw evaluation");

    parameters.quietComeback = 96;
    require(!Reveal::raw_eval_needed(parameters, noQuiet),
            "capture/empty context requested a quiet comeback evaluation");
    require(Reveal::raw_eval_needed(parameters, safeQuiet),
            "eligible quiet comeback did not request a raw evaluation");
    const int quietOnly = Reveal::combine_bonus(
      0, Reveal::quiet_bonus(parameters, {0, 1024, 0, 0, 1024}), safeQuiet.eligible);
    require(quietOnly == 96, "eligible quiet could not independently enable V2 reward");
    require(Reveal::combine_bonus(0, 96, noQuiet.eligible) == 0,
            "empty context admitted a V2 contribution");

    parameters               = {};
    parameters.bonusComeback = 64;
    require(Reveal::raw_eval_needed(parameters, noQuiet),
            "V1 comeback stopped requesting raw evaluation for a dark capture");
    parameters.quietComeback = 96;
    require(Reveal::raw_eval_needed(parameters, safeQuiet),
            "combined V1/V2 comeback did not share one raw-evaluation requirement");

    parameters               = {};
    parameters.quietBase     = 80;
    parameters.quietSafety   = 96;
    parameters.quietComeback = 0;
    require(!Reveal::raw_eval_needed(parameters, safeQuiet),
            "non-comeback quiet features triggered a lazy evaluation");

    parameters            = {};
    parameters.bonusBase  = 77;
    const int v1Only      = Reveal::bonus(parameters, {1024, 1024, 1024, 1024});
    const int combinedV1  = Reveal::combine_bonus(v1Only, 0, safeQuiet.eligible);
    require(combinedV1 == 77, "V1 nonzero plus V2 zero did not preserve V1 reward");

    parameters                 = {};
    parameters.bonusBase       = 256;
    parameters.bonusPhase      = 256;
    parameters.quietBase       = 96;
    parameters.quietSafety     = 192;
    const int clampedV1        = Reveal::bonus(parameters, {1024, 0, 0, 0});
    const int clampedV2        = Reveal::quiet_bonus(parameters, {0, 1024, 0, 0, 0});
    require(clampedV1 == 512 && clampedV2 == 192,
            "V1/V2 were not independently clamped before combination");
    require(Reveal::combine_bonus(clampedV1, clampedV2, safeQuiet.eligible) == 512,
            "final combined reward was not clamped after independent rewards");

    require(!Reveal::qsearch_quiet_context_enabled(true, false, true),
            "ordinary non-check qsearch enabled V2 quiet context");
    require(!Reveal::qsearch_quiet_context_enabled(false, true, true),
            "non-reveal qsearch enabled V2 quiet context");
    require(!Reveal::qsearch_quiet_context_enabled(true, true, false),
            "qsearch enabled an unnecessary quiet safety calculation");
    require(Reveal::qsearch_quiet_context_enabled(true, true, true),
            "in-check reveal qsearch did not enable its quiet evasion context");
}

void test_position_features_are_side_to_move_symmetric() {
    constexpr auto StartFEN = "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX";

    std::deque<StateInfo> states(2);
    Position              white;
    Position              black;
    const std::string     pools = "R2A2C2P5N2B2r2a2c2p5n2b2";
    white.set(std::string(StartFEN) + " w " + pools + " 0 1", &states[0]);
    black.set(std::string(StartFEN) + " b " + pools + " 0 1", &states[1]);

    const auto whiteFeatures = Reveal::make_features(white, Value(400));
    const auto blackFeatures = Reveal::make_features(black, Value(400));
    require(whiteFeatures.phase == 1024 && blackFeatures.phase == 1024,
            "position phase did not count both colors' dark pieces");
    require(whiteFeatures.pool == blackFeatures.pool,
            "identical color pools produced asymmetric pool features");
    require(whiteFeatures.unknown == 0 && blackFeatures.unknown == 0,
            "fully known starting pools produced unknown mass");
    require(whiteFeatures.comeback == -512 && blackFeatures.comeback == -512,
            "side-to-move raw evaluations produced asymmetric comeback features");

    const auto whiteQuiet =
      Reveal::make_quiet_features(white, Reveal::QuietSafety::Safe, Value(400));
    const auto blackQuiet =
      Reveal::make_quiet_features(black, Reveal::QuietSafety::Safe, Value(400));
    require(whiteQuiet.phase == 1024 && blackQuiet.phase == 1024,
            "quiet position phase did not count both colors' dark pieces");
    require(whiteQuiet.highValue == 0 && blackQuiet.highValue == 0
              && whiteQuiet.diversity == 0 && blackQuiet.diversity == 0,
            "initial quiet identity pools are not centered");
    require(whiteQuiet.safety == 1024 && blackQuiet.safety == 1024,
            "quiet position safety is asymmetric");
    require(whiteQuiet.comeback == -512 && blackQuiet.comeback == -512,
            "quiet position comeback is asymmetric");
}

void test_quiet_position_features_use_side_pool_and_ignore_counters() {
    constexpr auto Board = "4k4/9/9/x8/9/9/X8/9/9/4K4";

    std::deque<StateInfo> states(6);
    Position              rookWhite;
    Position              rookBlack;
    Position              pawnWhite;
    Position              pawnBlack;
    Position              baseCounters;
    Position              changedCounters;

    rookWhite.set(std::string(Board) + " w R1p1 0 1", &states[0]);
    rookBlack.set(std::string(Board) + " b R1p1 0 1", &states[1]);
    pawnWhite.set(std::string(Board) + " w P1r1 0 1", &states[2]);
    pawnBlack.set(std::string(Board) + " b P1r1 0 1", &states[3]);
    baseCounters.set(std::string(Board) + " w R1p1 0 1", &states[4]);
    changedCounters.set(std::string(Board) + " w R1p1 57 99", &states[5]);

    const auto rookWhiteFeatures =
      Reveal::make_quiet_features(rookWhite, Reveal::QuietSafety::Safe, VALUE_ZERO);
    const auto rookBlackFeatures =
      Reveal::make_quiet_features(rookBlack, Reveal::QuietSafety::Safe, VALUE_ZERO);
    require(rookWhiteFeatures.highValue == 1024,
            "R1p1 white quiet feature did not use the white rook pool");
    require(rookBlackFeatures.highValue == -1024,
            "R1p1 black quiet feature did not use the black pawn pool");

    const auto pawnWhiteFeatures =
      Reveal::make_quiet_features(pawnWhite, Reveal::QuietSafety::Safe, VALUE_ZERO);
    const auto pawnBlackFeatures =
      Reveal::make_quiet_features(pawnBlack, Reveal::QuietSafety::Safe, VALUE_ZERO);
    require(pawnWhiteFeatures.highValue == -1024,
            "P1r1 white quiet feature did not use the white pawn pool");
    require(pawnBlackFeatures.highValue == 1024,
            "P1r1 black quiet feature did not use the black rook pool");

    const auto base =
      Reveal::make_quiet_features(baseCounters, Reveal::QuietSafety::Safe, Value(400));
    const auto changed =
      Reveal::make_quiet_features(changedCounters, Reveal::QuietSafety::Safe, Value(400));
    require(base.phase == changed.phase && base.safety == changed.safety
              && base.highValue == changed.highValue && base.diversity == changed.diversity
              && base.comeback == changed.comeback,
            "quiet position features depend on FEN halfmove/fullmove counters");
}

void test_reward_transform_and_generalized_inverse() {
    for (Value value : {Value(-1000), VALUE_ZERO, Value(1000), VALUE_MATED_IN_MAX_PLY,
                        VALUE_MATE_IN_MAX_PLY, Value(-VALUE_MATE), Value(VALUE_MATE)})
        require(Reveal::apply_bonus(value, 0) == value, "B=0 changed a score");

    require(Reveal::apply_bonus(VALUE_MATED_IN_MAX_PLY, 512) == VALUE_MATED_IN_MAX_PLY,
            "mated score changed");
    require(Reveal::apply_bonus(VALUE_MATE_IN_MAX_PLY, -512) == VALUE_MATE_IN_MAX_PLY,
            "mate score changed");
    require(Reveal::apply_bonus(VALUE_ZERO, 512) == 512, "+512 bonus was not applied");
    require(Reveal::apply_bonus(VALUE_ZERO, -512) == -512, "-512 bonus was not applied");

    for (int bonus : {-512, -37, 0, 41, 512})
    {
        Value previous = -VALUE_MATE;
        for (int raw = -VALUE_MATE; raw <= VALUE_MATE; ++raw)
        {
            const Value adjusted = Reveal::apply_bonus(Value(raw), bonus);
            require(adjusted >= previous, "reward transform is not monotone");
            previous = adjusted;
        }
    }

    constexpr Value lower = VALUE_MATED_IN_MAX_PLY;
    constexpr Value upper = VALUE_MATE_IN_MAX_PLY;
    require(Reveal::first_raw(lower, 512) == lower, "inverse changed lower decisive bound");
    require(Reveal::first_raw(upper, -512) == upper, "inverse changed upper decisive bound");
    require(Reveal::first_raw(lower + 1, -512) == lower + 1,
            "inverse mishandled lower non-decisive bound");
    require(Reveal::first_raw(100, 25) == 75, "positive inverse shift is wrong");
    require(Reveal::first_raw(100, -25) == 125, "negative inverse shift is wrong");
    require(Reveal::first_raw(upper - 1, -512) == upper,
            "unreachable adjusted threshold did not map to upper decisive bound");

    const auto identityWindow = Reveal::child_window(-20, 40, 0);
    require(identityWindow.alpha == -40 && identityWindow.beta == 20,
            "B=0 child window differs from [-beta,-alpha]");

    const auto shiftedWindow = Reveal::child_window(-20, 40, 15);
    require(shiftedWindow.alpha == -25 && shiftedWindow.beta == 35,
            "bonus-aware child window is incorrect");
}

void test_wrapper_applies_bonus_once_to_aggregated_result() {
    int   calls      = 0;
    Value childAlpha = VALUE_NONE;
    Value childBeta  = VALUE_NONE;

    const Value result = Reveal::search_with_bonus(-20, 40, 15, [&](Value alpha, Value beta) {
        ++calls;
        childAlpha = alpha;
        childBeta  = beta;
        return Value(-25);  // Represents the already aggregated hidden-identity result.
    });

    require(calls == 1, "bonus wrapper searched the identity aggregate more than once");
    require(childAlpha == -25 && childBeta == 35, "wrapper passed the wrong child window");
    require(result == 40, "wrapper did not apply the bonus exactly once");
    require(result != Reveal::apply_bonus(Reveal::apply_bonus(25, 15), 15),
            "wrapper result matches a double-applied bonus");
}

void test_pruning_and_move_order_helpers_preserve_zero_path() {
    require(Reveal::pruning_slots(0) == 0, "P=0 produced extra move slots");
    require(Reveal::pruning_slots(1) == 1 && Reveal::pruning_slots(64) == 1
              && Reveal::pruning_slots(65) == 2,
            "positive pruning slots did not use ceil(P/64)");
    require(Reveal::pruning_slots(-1) == -1 && Reveal::pruning_slots(-64) == -1
              && Reveal::pruning_slots(-65) == -2,
            "negative pruning slots did not use signed ceil(abs(P)/64)");

    for (int raw : {-20000, -1, 0, 1, 20000})
    {
        require(Reveal::move_order_score(raw, false, 1234) == raw,
                "visible move received reveal ordering");
        require(Reveal::move_order_score(raw, true, 0) == raw,
                "zero move ordering changed a dark score");
        const int adjusted = Reveal::move_order_score(raw, true, 1234);
        require(adjusted == raw + 1234, "dark move ordering was not added");
        require(Reveal::raw_move_score(adjusted, true, 1234) == raw,
                "raw bucket score did not remove reveal ordering");
    }
}

void test_v2_ordering_helpers_are_selective_and_reversible() {
    Reveal::OrderingParameters ordering{4096, 1024, 0};
    constexpr int              raw = -14001;

    require(Reveal::ordered_score(raw, false, false, ordering) == raw,
            "visible move received reveal ordering");
    require(Reveal::ordered_score(raw, true, false, ordering) == raw + 4096,
            "dark move did not receive V1 ordering");
    require(Reveal::ordered_score(raw, true, true, ordering) == raw + 4096 + 1024,
            "safe dark quiet did not receive both ordering terms");
    require(Reveal::raw_score(
              Reveal::ordered_score(raw, true, true, ordering), true, true, ordering)
              == raw,
            "safe quiet ordering adjustment was not reversible");

    ordering.quietMoveOrder = -2048;
    require(Reveal::ordered_score(raw, true, true, ordering) == raw + 4096 - 2048,
            "negative quiet ordering was not applied");
    require(Reveal::raw_score(
              Reveal::ordered_score(raw, true, true, ordering), true, true, ordering)
              == raw,
            "negative quiet ordering was not reversible");

    ordering.pruningMargin = 512;
    require(Reveal::ordered_score(raw, true, true, ordering) == raw + 4096 - 2048,
            "pruning margin leaked into move ordering");

    constexpr Reveal::OrderingParameters zero{};
    for (int score : {-20000, -1, 0, 1, 20000})
    {
        require(Reveal::ordered_score(score, false, false, zero) == score,
                "zero ordering changed a visible score");
        require(Reveal::ordered_score(score, true, true, zero) == score,
                "zero ordering changed a safe dark quiet score");
        require(Reveal::raw_score(score, true, true, zero) == score,
                "zero ordering changed a recovered raw score");
    }
}

struct MovePickerHistories {
    std::unique_ptr<ButterflyHistory>      main = std::make_unique<ButterflyHistory>();
    std::unique_ptr<LowPlyHistory>         lowPly = std::make_unique<LowPlyHistory>();
    std::unique_ptr<CapturePieceToHistory> capture =
      std::make_unique<CapturePieceToHistory>();
    std::array<std::unique_ptr<PieceToHistory>, 6> continuation;
    std::array<const PieceToHistory*, 6>           continuationPointers{};
    std::unique_ptr<PawnHistory>                   pawn = std::make_unique<PawnHistory>();

    MovePickerHistories() {
        main->fill(0);
        lowPly->fill(0);
        capture->fill(0);
        pawn->fill(0);
        for (std::size_t i = 0; i < continuation.size(); ++i)
        {
            continuation[i] = std::make_unique<PieceToHistory>();
            continuation[i]->fill(0);
            continuationPointers[i] = continuation[i].get();
        }
    }
};

Move checked_picker_move(Position& pos,
                         const char* uci,
                         bool        dark,
                         bool        capture,
                         const char* fixtureName) {
    const Move move = UCIEngine::to_move(pos, uci);
    const std::string context = std::string(fixtureName) + " " + uci;
    require(move.is_ok(), context + " is unavailable");
    require(pos.pseudo_legal(move), context + " is not pseudo-legal");
    require(pos.legal(move), context + " is not legal");
    require(pos.move_dark(move) == dark, context + " has the wrong dark property");
    require(pos.capture(move) == capture, context + " has the wrong capture property");
    return move;
}

void test_quiet_move_context_is_selective_safe_and_counter_independent() {
    struct Fixture {
        const char*         name;
        const char*         fen;
        Reveal::QuietSafety safety;
    };
    constexpr Fixture fixtures[] = {
      {"safe context", "4k4/9/9/9/4P4/9/X8/9/9/R3K4 w P1 0 1",
       Reveal::QuietSafety::Safe},
      {"contested context", "4k4/9/9/9/4P4/8r/X8/9/9/R3K4 w P1 0 1",
       Reveal::QuietSafety::Contested},
      {"loose context", "4k4/9/9/9/4P4/8r/X8/9/9/4K4 w P1 0 1",
       Reveal::QuietSafety::Loose}};

    for (const Fixture& fixture : fixtures)
    {
        StateInfo state;
        Position  pos;
        pos.set(fixture.fen, &state);
        const Move darkQuiet =
          checked_picker_move(pos, "a3a4", true, false, fixture.name);

        const auto disabled = Reveal::quiet_move_context(pos, darkQuiet, false);
        require(!disabled.eligible && !disabled.safe(),
                std::string(fixture.name) + " remained eligible while disabled");

        const auto context = Reveal::quiet_move_context(pos, darkQuiet, true);
        require(context.eligible, std::string(fixture.name) + " was not eligible");
        require(context.safety == fixture.safety,
                std::string(fixture.name) + " received the wrong safety class");
        require(context.safe() == (fixture.safety == Reveal::QuietSafety::Safe),
                std::string(fixture.name) + " safe() gate is wrong");
    }

    StateInfo visibleState;
    Position  visiblePosition;
    visiblePosition.set(fixtures[0].fen, &visibleState);
    const Move visibleQuiet =
      checked_picker_move(visiblePosition, "e5e6", false, false, "visible quiet context");
    const auto visible = Reveal::quiet_move_context(visiblePosition, visibleQuiet, true);
    require(!visible.eligible && !visible.safe(), "visible quiet received a V2 context");

    constexpr auto DarkCaptureFEN = "4k4/9/9/4p4/4P4/p8/X8/9/9/4K4 w P1 0 1";
    StateInfo     captureState;
    Position      capturePosition;
    capturePosition.set(DarkCaptureFEN, &captureState);
    const Move darkCapture =
      checked_picker_move(capturePosition, "a3a4", true, true, "dark capture context");
    const auto capture = Reveal::quiet_move_context(capturePosition, darkCapture, true);
    require(!capture.eligible && !capture.safe(), "dark capture received a V2 quiet context");

    constexpr auto CounterBoard = "4k4/9/9/9/4P4/9/X8/9/9/R3K4 w P1";
    StateInfo     baseState;
    StateInfo     changedState;
    Position      baseCounters;
    Position      changedCounters;
    baseCounters.set(std::string(CounterBoard) + " 0 1", &baseState);
    changedCounters.set(std::string(CounterBoard) + " 37 88", &changedState);
    const Move baseMove =
      checked_picker_move(baseCounters, "a3a4", true, false, "base-counter context");
    const Move changedMove =
      checked_picker_move(changedCounters, "a3a4", true, false, "changed-counter context");
    const auto baseContext = Reveal::quiet_move_context(baseCounters, baseMove, true);
    const auto changedContext = Reveal::quiet_move_context(changedCounters, changedMove, true);
    require(baseContext.eligible == changedContext.eligible
              && baseContext.safety == changedContext.safety
              && baseContext.safe() == changedContext.safe(),
            "quiet move context depends on FEN halfmove/fullmove counters");
}

void test_reveal_reduction_combines_v1_and_safe_only_v2_without_ply_scaling() {
    Reveal::Parameters parameters{};
    parameters.reduction      = 300;
    parameters.quietReduction = 545;

    require(Reveal::adjust_reduction(Depth(2000), true, true, parameters) == 1155,
            "V1 and safe-quiet reductions were not both subtracted in raw units");
    require(Reveal::adjust_reduction(Depth(2000), true, false, parameters) == 1700,
            "non-safe dark move did not receive exactly the V1 reduction");
    require(Reveal::adjust_reduction(Depth(2000), false, false, parameters) == 2000,
            "visible move received a reveal reduction");

    parameters.quietReduction = -545;
    require(Reveal::adjust_reduction(Depth(2000), true, true, parameters) == 2245,
            "negative safe-quiet reduction endpoint was clamped, scaled, or signed incorrectly");

    constexpr auto ContestedFEN = "4k4/9/9/9/4P4/8r/X8/9/9/R3K4 w P1 0 1";
    constexpr auto LooseFEN     = "4k4/9/9/9/4P4/8r/X8/9/9/4K4 w P1 0 1";
    constexpr auto CaptureFEN   = "4k4/9/9/4p4/4P4/p8/X8/9/9/4K4 w P1 0 1";
    std::array<StateInfo, 3> states;
    Position                 contested;
    Position                 loose;
    Position                 capture;
    contested.set(ContestedFEN, &states[0]);
    loose.set(LooseFEN, &states[1]);
    capture.set(CaptureFEN, &states[2]);
    const Move contestedMove =
      checked_picker_move(contested, "a3a4", true, false, "contested reduction");
    const Move looseMove = checked_picker_move(loose, "a3a4", true, false, "loose reduction");
    const Move captureMove =
      checked_picker_move(capture, "a3a4", true, true, "capture reduction");

    parameters.quietReduction = 545;
    require(Reveal::adjust_reduction(
              Depth(2000), true, Reveal::quiet_move_context(contested, contestedMove, true).safe(),
              parameters)
              == 1700,
            "contested quiet received the safe-only reduction");
    require(Reveal::adjust_reduction(
              Depth(2000), true, Reveal::quiet_move_context(loose, looseMove, true).safe(), parameters)
              == 1700,
            "loose quiet received the safe-only reduction");
    require(Reveal::adjust_reduction(
              Depth(2000), true, Reveal::quiet_move_context(capture, captureMove, true).safe(),
              parameters)
              == 1700,
            "dark capture lost V1 reduction or received V2 reduction");

    parameters = {};
    require(Reveal::adjust_reduction(Depth(-700), true, true, parameters) == -700,
            "all-zero reveal reductions changed a negative raw reduction");
}

Move first_main_move(const Position&                pos,
                     Move                           ttMove,
                     MovePickerHistories&           histories,
                     Reveal::OrderingParameters     ordering) {
    MovePicker picker(pos, ttMove, Depth(1), histories.main.get(), histories.lowPly.get(),
                      histories.capture.get(), histories.continuationPointers.data(),
                      histories.pawn.get(), 0, ordering);
    return picker.next_reveal_move(false, false);
}

Move first_main_move_skipping_visible_quiets(const Position&            pos,
                                              MovePickerHistories&       histories,
                                              Reveal::OrderingParameters ordering) {
    MovePicker picker(pos, Move::none(), Depth(1), histories.main.get(),
                      histories.lowPly.get(), histories.capture.get(),
                      histories.continuationPointers.data(), histories.pawn.get(), 0, ordering);
    picker.skip_quiet_moves();
    return picker.next_reveal_move(true, false);
}

Move first_probcut_move(const Position&            pos,
                        Move                       ttMove,
                        int                        threshold,
                        MovePickerHistories&       histories,
                        Reveal::OrderingParameters ordering) {
    MovePicker picker(pos, ttMove, threshold, histories.capture.get(), ordering);
    return picker.next_reveal_move(false, false);
}

void test_move_picker_orders_only_safe_dark_quiets_and_preserves_tt_priority() {
    struct Fixture {
        const char*         name;
        const char*         fen;
        Reveal::QuietSafety safety;
    };
    constexpr Fixture fixtures[] = {
      {"safe", "4k4/9/9/9/4P4/9/X8/9/9/R3K4 w P1 0 1", Reveal::QuietSafety::Safe},
      {"contested", "4k4/9/9/9/4P4/8r/X8/9/9/R3K4 w P1 0 1",
       Reveal::QuietSafety::Contested},
      {"loose", "4k4/9/9/9/4P4/8r/X8/9/9/4K4 w P1 0 1", Reveal::QuietSafety::Loose}};

    for (const Fixture& fixture : fixtures)
    {
        StateInfo state;
        Position  pos;
        pos.set(fixture.fen, &state);
        const Move darkQuiet = checked_picker_move(pos, "a3a4", true, false, fixture.name);
        const Move visible = checked_picker_move(pos, "e5e6", false, false, fixture.name);
        require(Reveal::quiet_safety(pos, darkQuiet) == fixture.safety,
                std::string(fixture.name) + " fixture has the wrong safety class");

        MovePickerHistories histories;
        (*histories.main)[pos.side_to_move()][visible.from_to()] = 500;  // raw = 1000

        const Move first = first_main_move(pos, Move::none(), histories, {0, 2048, 0});
        require(first == (fixture.safety == Reveal::QuietSafety::Safe ? darkQuiet : visible),
                std::string(fixture.name)
                  + " quiet ordering did not distinguish safe from non-safe dark quiets");
    }

    StateInfo contestedState;
    Position  contested;
    contested.set(fixtures[1].fen, &contestedState);
    const Move contestedDark =
      checked_picker_move(contested, "a3a4", true, false, "V1 dark quiet");
    const Move contestedVisible =
      checked_picker_move(contested, "e5e6", false, false, "V1 visible quiet");
    MovePickerHistories v1Histories;
    (*v1Histories.main)[contested.side_to_move()][contestedVisible.from_to()] = 500;
    require(first_main_move(contested, Move::none(), v1Histories, {2048, 0, 0})
              == contestedDark,
            "V1 ordering no longer applies to a non-safe dark quiet");

    StateInfo safeState;
    Position  safe;
    safe.set(fixtures[0].fen, &safeState);
    const Move safeDark = checked_picker_move(safe, "a3a4", true, false, "TT dark quiet");
    const Move safeVisible =
      checked_picker_move(safe, "e5e6", false, false, "TT visible quiet");
    MovePickerHistories positiveHistories;
    require(first_main_move(safe, safeVisible, positiveHistories, {16384, 2048, 0})
              == safeVisible,
            "positive reveal ordering displaced the TT move");
    MovePickerHistories negativeHistories;
    require(first_main_move(safe, safeDark, negativeHistories, {-16384, -2048, 0}) == safeDark,
            "negative reveal ordering displaced the TT move");
}

void test_move_picker_keeps_v2_off_captures_and_v1_on_dark_captures() {
    constexpr auto fen = "4k4/9/9/4p4/4P4/p8/X8/9/9/4K4 w P1 0 1";
    StateInfo     state;
    Position      pos;
    pos.set(fen, &state);
    const Move darkCapture =
      checked_picker_move(pos, "a3a4", true, true, "dark capture ordering");
    const Move visibleCapture =
      checked_picker_move(pos, "e5e6", false, true, "visible capture ordering");

    MovePickerHistories v2Histories;
    (*v2Histories.capture)[pos.moved_piece(visibleCapture)][visibleCapture.to_sq()]
                          [type_of(pos.piece_on(visibleCapture.to_sq()))] = 1000;
    require(first_main_move(pos, Move::none(), v2Histories, {0, 2048, 0}) == visibleCapture,
            "V2 quiet ordering leaked onto a dark capture");

    MovePickerHistories v1Histories;
    (*v1Histories.capture)[pos.moved_piece(visibleCapture)][visibleCapture.to_sq()]
                          [type_of(pos.piece_on(visibleCapture.to_sq()))] = 1000;
    require(first_main_move(pos, Move::none(), v1Histories, {2048, 0, 0}) == darkCapture,
            "V1 dark ordering no longer applies to a dark capture");
}

void test_good_quiet_bucket_uses_raw_score_after_safe_ordering() {
    constexpr auto fen = "4k4/9/9/4p4/4P4/9/X8/9/9/4K4 w P1 0 1";
    StateInfo     state;
    Position      pos;
    pos.set(fen, &state);
    const Move darkQuiet =
      checked_picker_move(pos, "a3a4", true, false, "GOOD_QUIET raw bucket");
    const Move visibleCapture =
      checked_picker_move(pos, "e5e6", false, true, "BAD_CAPTURE sentinel");
    require(Reveal::quiet_safety(pos, darkQuiet) == Reveal::QuietSafety::Safe,
            "GOOD_QUIET raw bucket fixture is not safe");
    require(!pos.see_ge(visibleCapture, 400),
            "capture sentinel is not sufficiently losing for the BAD_CAPTURE bucket");

    MovePickerHistories histories;
    (*histories.main)[pos.side_to_move()][darkQuiet.from_to()] = -7001;  // raw = -14002
    (*histories.capture)[pos.moved_piece(visibleCapture)][visibleCapture.to_sq()]
                        [type_of(pos.piece_on(visibleCapture.to_sq()))] = -10000;

    require(first_main_move_skipping_visible_quiets(pos, histories, {0, 2048, 0})
              == visibleCapture,
            "stored safe-quiet ordering score changed the GOOD_QUIET raw bucket");
}

void test_good_capture_see_uses_raw_score_after_dark_ordering() {
    constexpr auto fen = "4k4/9/9/9/4P4/p8/X8/9/9/4K4 w P1 0 1";
    StateInfo     state;
    Position      pos;
    pos.set(fen, &state);
    const Move darkCapture =
      checked_picker_move(pos, "a3a4", true, true, "GOOD_CAPTURE raw SEE");
    const Move visibleQuiet =
      checked_picker_move(pos, "e5e6", false, false, "GOOD_QUIET sentinel");
    require(!pos.see_ge(darkCapture, 400),
            "dark capture is not sufficiently losing for the GOOD_CAPTURE boundary");

    MovePickerHistories histories;
    (*histories.capture)[pos.moved_piece(darkCapture)][darkCapture.to_sq()]
                        [type_of(pos.piece_on(darkCapture.to_sq()))] = -10000;
    (*histories.main)[pos.side_to_move()][visibleQuiet.from_to()] = 2000;  // raw = 4000

    require(first_main_move(pos, Move::none(), histories, {16384, 2048, 0}) == visibleQuiet,
            "stored dark ordering score changed the GOOD_CAPTURE SEE bucket");
}

void test_probcut_qualification_uses_only_see_and_pruning_margin() {
    constexpr auto fen = "4k4/9/9/9/4P4/p8/X8/9/9/4K4 w P1 0 1";
    StateInfo     state;
    Position      pos;
    pos.set(fen, &state);
    const Move darkCapture =
      checked_picker_move(pos, "a3a4", true, true, "ProbCut dark capture");
    require(!pos.see_ge(darkCapture, 200) && pos.see_ge(darkCapture, 136),
            "ProbCut fixture does not straddle threshold 200 with margin 64");

    MovePickerHistories positiveOrder;
    require(first_probcut_move(pos, Move::none(), 200, positiveOrder, {16384, 2048, 0})
              == Move::none(),
            "positive ordering changed ProbCut SEE qualification");
    MovePickerHistories negativeOrder;
    require(first_probcut_move(pos, Move::none(), 200, negativeOrder, {-16384, -2048, 0})
              == Move::none(),
            "negative ordering changed ProbCut SEE qualification");

    MovePickerHistories margin;
    require(first_probcut_move(pos, Move::none(), 200, margin, {16384, 2048, 64})
              == darkCapture,
            "pruning margin did not relax dark ProbCut SEE to threshold-P");

    MovePickerHistories rejectedTt;
    require(first_probcut_move(pos, darkCapture, 200, rejectedTt, {16384, 2048, 0})
              == Move::none(),
            "ordering made an ineligible ProbCut TT move eligible");
    MovePickerHistories eligibleTt;
    require(first_probcut_move(pos, darkCapture, 200, eligibleTt, {-16384, -2048, 64})
              == darkCapture,
            "eligible ProbCut TT move lost priority under negative ordering");
}

void test_quiet_pruning_limits_apply_after_the_current_move() {
    constexpr int base = 5;

    auto state = Reveal::quiet_pruning_after_move(base - 1, base, 2);
    require(!state.skipVisibleQuiets && !state.skipDarkQuiets,
            "positive slots pruned a quiet before either limit");

    state = Reveal::quiet_pruning_after_move(base, base, 2);
    require(state.skipVisibleQuiets && !state.skipDarkQuiets,
            "positive slots did not preserve dark quiets after the visible limit");

    state = Reveal::quiet_pruning_after_move(base + 1, base, 2);
    require(state.skipVisibleQuiets && !state.skipDarkQuiets,
            "positive slots pruned dark quiets before their extended limit");

    state = Reveal::quiet_pruning_after_move(base + 2, base, 2);
    require(state.skipVisibleQuiets && state.skipDarkQuiets,
            "positive slots did not prune both quiet classes at their limits");

    state = Reveal::quiet_pruning_after_move(base - 3, base, -2);
    require(!state.skipVisibleQuiets && !state.skipDarkQuiets,
            "negative slots pruned a quiet before either limit");

    state = Reveal::quiet_pruning_after_move(base - 2, base, -2);
    require(!state.skipVisibleQuiets && state.skipDarkQuiets,
            "negative slots did not prune dark quiets at their shortened limit");

    state = Reveal::quiet_pruning_after_move(base - 1, base, -2);
    require(!state.skipVisibleQuiets && state.skipDarkQuiets,
            "negative slots pruned visible quiets before the base limit");

    state = Reveal::quiet_pruning_after_move(base, base, -2);
    require(state.skipVisibleQuiets && state.skipDarkQuiets,
            "negative slots did not prune both quiet classes at their limits");
}

void test_research_raw_eval_uses_current_uncorrected_tt_value_only() {
    require(Reveal::raw_eval_for_research(Value(137), Value(-91)) == Value(137),
            "excluded research used correction-adjusted static eval instead of current raw TT eval");
    require(Reveal::raw_eval_for_research(VALUE_NONE, Value(-91)) == VALUE_NONE,
            "excluded research treated correction-adjusted static eval as a raw fallback");
}

void set_option(Engine& engine, const std::string& name, int value) {
    std::istringstream command("name " + name + " value " + std::to_string(value));
    engine.get_options().setoption(command);
}

struct QuietOptionSpec {
    const char* name;
    int         min;
    int         max;
    int         defaultValue;
};

constexpr std::array<QuietOptionSpec, 8> QuietOptionSpecs = {
  {{"RevealQuietBase", -96, 96, -12},
   {"RevealQuietPhase", -96, 96, 0},
   {"RevealQuietSafety", -192, 192, 24},
   {"RevealQuietHighValue", -96, 96, -12},
   {"RevealQuietDiversity", -96, 96, 0},
   {"RevealQuietComeback", -96, 96, 0},
   {"RevealQuietMoveOrder", -2048, 2048, 0},
   {"RevealQuietReduction", -545, 545, 0}}};

void test_reveal_uci_defaults_and_ranges(Engine& engine) {
    auto& options = engine.get_options();

    std::ostringstream listing;
    listing << options;
    const std::string text = listing.str();

    const std::string valueOptions[] = {"RevealBonusBase", "RevealBonusPhase", "RevealBonusPool",
                                        "RevealBonusUnknown", "RevealBonusComeback"};
    for (const std::string& name : valueOptions)
    {
        require(options.count(name) == 1, name + " was not registered");
        require(int(options[name]) == 0, name + " default was not zero");
        require(text.find("option name " + name + " type spin default 0 min -256 max 256")
                  != std::string::npos,
                name + " range is incorrect");
    }

    require(text.find("option name RevealMoveOrder type spin default 0 min -16384 max 16384")
              != std::string::npos,
            "RevealMoveOrder range is incorrect");
    require(text.find("option name RevealReduction type spin default 0 min -2180 max 2180")
              != std::string::npos,
            "RevealReduction range is incorrect");
    require(text.find("option name RevealPruningMargin type spin default 0 min -512 max 512")
              != std::string::npos,
            "RevealPruningMargin range is incorrect");

    const std::string revealOptionBlock =
      "\noption name RevealBonusBase type spin default 0 min -256 max 256"
      "\noption name RevealBonusPhase type spin default 0 min -256 max 256"
      "\noption name RevealBonusPool type spin default 0 min -256 max 256"
      "\noption name RevealBonusUnknown type spin default 0 min -256 max 256"
      "\noption name RevealBonusComeback type spin default 0 min -256 max 256"
      "\noption name RevealMoveOrder type spin default 0 min -16384 max 16384"
      "\noption name RevealReduction type spin default 0 min -2180 max 2180"
      "\noption name RevealPruningMargin type spin default 0 min -512 max 512"
      "\noption name RevealQuietBase type spin default -12 min -96 max 96"
      "\noption name RevealQuietPhase type spin default 0 min -96 max 96"
      "\noption name RevealQuietSafety type spin default 24 min -192 max 192"
      "\noption name RevealQuietHighValue type spin default -12 min -96 max 96"
      "\noption name RevealQuietDiversity type spin default 0 min -96 max 96"
      "\noption name RevealQuietComeback type spin default 0 min -96 max 96"
      "\noption name RevealQuietMoveOrder type spin default 0 min -2048 max 2048"
      "\noption name RevealQuietReduction type spin default 0 min -545 max 545";
    require(text.find(revealOptionBlock) != std::string::npos,
            "V1/V2 reveal UCI options are missing, out of order, or not consecutive");

    for (const QuietOptionSpec& spec : QuietOptionSpecs)
    {
        require(options.count(spec.name) == 1, std::string(spec.name) + " was not registered");
        require(int(options[spec.name]) == spec.defaultValue,
                std::string(spec.name) + " tuned default is wrong");

        set_option(engine, spec.name, spec.min);
        require(int(options[spec.name]) == spec.min,
                std::string(spec.name) + " lower bound was rejected");
        set_option(engine, spec.name, spec.min - 1);
        require(int(options[spec.name]) == spec.min,
                std::string(spec.name) + " accepted a value below its range");

        set_option(engine, spec.name, spec.max);
        require(int(options[spec.name]) == spec.max,
                std::string(spec.name) + " upper bound was rejected");
        set_option(engine, spec.name, spec.max + 1);
        require(int(options[spec.name]) == spec.max,
                std::string(spec.name) + " accepted a value above its range");
        set_option(engine, spec.name, 0);
    }

    set_option(engine, "RevealBonusBase", 256);
    require(int(options["RevealBonusBase"]) == 256, "RevealBonusBase upper bound was rejected");
    set_option(engine, "RevealBonusBase", 257);
    require(int(options["RevealBonusBase"]) == 256,
            "RevealBonusBase accepted a value above its range");
    set_option(engine, "RevealMoveOrder", -16384);
    require(int(options["RevealMoveOrder"]) == -16384, "RevealMoveOrder lower bound was rejected");
    set_option(engine, "RevealReduction", 2180);
    require(int(options["RevealReduction"]) == 2180, "RevealReduction upper bound was rejected");
    set_option(engine, "RevealPruningMargin", -512);
    require(int(options["RevealPruningMargin"]) == -512,
            "RevealPruningMargin lower bound was rejected");
}

void test_reveal_worker_parameter_snapshot(Engine& engine) {
    set_option(engine, "RevealBonusBase", 11);
    set_option(engine, "RevealBonusPhase", 12);
    set_option(engine, "RevealBonusPool", 13);
    set_option(engine, "RevealBonusUnknown", 14);
    set_option(engine, "RevealBonusComeback", 15);
    set_option(engine, "RevealMoveOrder", 16);
    set_option(engine, "RevealReduction", 17);
    set_option(engine, "RevealPruningMargin", 18);
    set_option(engine, "RevealQuietBase", 21);
    set_option(engine, "RevealQuietPhase", 22);
    set_option(engine, "RevealQuietSafety", 23);
    set_option(engine, "RevealQuietHighValue", 24);
    set_option(engine, "RevealQuietDiversity", 25);
    set_option(engine, "RevealQuietComeback", 26);
    set_option(engine, "RevealQuietMoveOrder", 27);
    set_option(engine, "RevealQuietReduction", 28);

    const Reveal::Parameters parameters =
      Search::snapshot_reveal_parameters(engine.get_options());
    require(parameters.bonusBase == 11, "RevealBonusBase cache is wrong");
    require(parameters.bonusPhase == 12, "RevealBonusPhase cache is wrong");
    require(parameters.bonusPool == 13, "RevealBonusPool cache is wrong");
    require(parameters.bonusUnknown == 14, "RevealBonusUnknown cache is wrong");
    require(parameters.bonusComeback == 15, "RevealBonusComeback cache is wrong");
    require(parameters.moveOrder == 16, "RevealMoveOrder cache is wrong");
    require(parameters.reduction == 17, "RevealReduction cache is wrong");
    require(parameters.pruningMargin == 18, "RevealPruningMargin cache is wrong");
    require(parameters.quietBase == 21, "RevealQuietBase cache is wrong");
    require(parameters.quietPhase == 22, "RevealQuietPhase cache is wrong");
    require(parameters.quietSafety == 23, "RevealQuietSafety cache is wrong");
    require(parameters.quietHighValue == 24, "RevealQuietHighValue cache is wrong");
    require(parameters.quietDiversity == 25, "RevealQuietDiversity cache is wrong");
    require(parameters.quietComeback == 26, "RevealQuietComeback cache is wrong");
    require(parameters.quietMoveOrder == 27, "RevealQuietMoveOrder cache is wrong");
    require(parameters.quietReduction == 28, "RevealQuietReduction cache is wrong");

    constexpr std::array<const char*, 16> allRevealOptions = {
      "RevealBonusBase",       "RevealBonusPhase",      "RevealBonusPool",
      "RevealBonusUnknown",    "RevealBonusComeback",   "RevealMoveOrder",
      "RevealReduction",       "RevealPruningMargin",   "RevealQuietBase",
      "RevealQuietPhase",      "RevealQuietSafety",     "RevealQuietHighValue",
      "RevealQuietDiversity",  "RevealQuietComeback",   "RevealQuietMoveOrder",
      "RevealQuietReduction"};
    for (const char* name : allRevealOptions)
        set_option(engine, name, 0);
}

struct SearchSignature {
    std::string bestMove;
    std::string score;
    size_t      nodes   = 0;
    bool        sawInfo = false;
};

SearchSignature search_signature(Engine&             engine,
                                 const std::string&   fen,
                                 int                  depth,
                                 const std::string&   searchMove = {}) {
    SearchSignature signature;
    engine.set_on_update_no_moves([](const Engine::InfoShort&) { });
    engine.set_on_iter([](const Engine::InfoIter&) { });
    engine.set_on_update_full([&](const Engine::InfoFull& info) {
        signature.score   = UCIEngine::format_score(info.score);
        signature.nodes   = info.nodes;
        signature.sawInfo = true;
    });
    engine.set_on_bestmove(
      [&](std::string_view move, std::string_view) { signature.bestMove = move; });

    engine.set_position(fen, {});
    Search::LimitsType limits;
    limits.depth = depth;
    if (!searchMove.empty())
        limits.searchmoves = {searchMove};
    engine.go(limits);
    engine.wait_for_search_finished();

    require(signature.sawInfo, "clearSearch fixture did not report a full info line");
    require(!signature.bestMove.empty(), "clearSearch fixture did not report bestmove");
    return signature;
}

SearchSignature fresh_search_signature(Engine&             engine,
                                       const std::string&   fen,
                                       int                  depth,
                                       const std::string&   searchMove = {}) {
    engine.search_clear();
    return search_signature(engine, fen, depth, searchMove);
}

int centipawn_score(const SearchSignature& signature, const char* context) {
    constexpr std::string_view Prefix = "cp ";
    require(signature.score.compare(0, Prefix.size(), Prefix) == 0,
            std::string(context) + " unexpectedly produced " + signature.score);
    return std::stoi(signature.score.substr(Prefix.size()));
}

bool uci_score_matches_bonus(const std::string&     fen,
                             const SearchSignature& baseline,
                             const SearchSignature& adjusted,
                             int                    bonus,
                             int                    applications) {
    StateInfo state;
    Position  pos;
    pos.set(fen, &state);
    const int baselineCp = centipawn_score(baseline, "bonus baseline");
    const int adjustedCp = centipawn_score(adjusted, "bonus result");

    for (int raw = VALUE_MATED_IN_MAX_PLY + 1; raw < VALUE_MATE_IN_MAX_PLY; ++raw)
    {
        if (UCIEngine::to_cp(Value(raw), pos) != baselineCp)
            continue;

        Value value = Value(raw);
        for (int i = 0; i < applications; ++i)
            value = Reveal::apply_bonus(value, bonus);
        if (UCIEngine::to_cp(value, pos) == adjustedCp)
            return true;
    }
    return false;
}

void reset_reveal_options(Engine& engine) {
    constexpr std::array<const char*, 16> Names = {
      "RevealBonusBase",       "RevealBonusPhase",      "RevealBonusPool",
      "RevealBonusUnknown",    "RevealBonusComeback",   "RevealMoveOrder",
      "RevealReduction",       "RevealPruningMargin",   "RevealQuietBase",
      "RevealQuietPhase",      "RevealQuietSafety",     "RevealQuietHighValue",
      "RevealQuietDiversity",  "RevealQuietComeback",   "RevealQuietMoveOrder",
      "RevealQuietReduction"};
    for (const char* name : Names)
        set_option(engine, name, 0);
}

bool same_signature(const SearchSignature& lhs, const SearchSignature& rhs) {
    return lhs.bestMove == rhs.bestMove && lhs.score == rhs.score && lhs.nodes == rhs.nodes;
}

void test_reveal_quiet_options_clear_search(Engine& engine) {
    constexpr auto FEN =
      "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w "
      "R2A2C2P5N2B2r2a2c2p5n2b2 0 1";

    set_option(engine, "Threads", 1);
    set_option(engine, "Hash", 1);
    for (const QuietOptionSpec& spec : QuietOptionSpecs)
        set_option(engine, spec.name, 0);

    engine.search_clear();
    const SearchSignature cold = search_signature(engine, FEN, 4);
    const SearchSignature warm = search_signature(engine, FEN, 4);
    require(!same_signature(cold, warm),
            "clearSearch fixture has no observable cold/warm TT distinction");

    for (const QuietOptionSpec& spec : QuietOptionSpecs)
    {
        set_option(engine, spec.name, 1);
        set_option(engine, spec.name, 0);
        const SearchSignature cleared = search_signature(engine, FEN, 4);
        require(cleared.bestMove == cold.bestMove,
                std::string(spec.name) + " callback did not restore the cold bestmove");
        require(cleared.score == cold.score,
                std::string(spec.name) + " callback did not restore the cold score");
        require(cleared.nodes == cold.nodes,
                std::string(spec.name) + " callback did not restore the cold node count");
    }
}

std::string
search_once(Engine& engine, const std::string& fen, int depth, const std::string& searchMove = {}) {
    std::string bestMove;
    engine.set_position(fen, {});
    const std::string before = engine.fen();
    engine.set_on_bestmove([&](std::string_view move, std::string_view) { bestMove = move; });

    Search::LimitsType limits;
    limits.depth = depth;
    if (!searchMove.empty())
        limits.searchmoves = {searchMove};
    engine.go(limits);
    engine.wait_for_search_finished();

    require(!bestMove.empty(), "reveal search did not report a best move");
    require(engine.fen() == before, "reveal search changed the root position or identity pool");
    if (!searchMove.empty())
        require(bestMove.compare(0, 4, searchMove) == 0, "forced reveal move was not searched");
    return bestMove;
}

void test_reveal_search_paths_and_in_check_comeback(Engine& engine) {
    engine.set_on_update_no_moves([](const Engine::InfoShort&) { });
    engine.set_on_update_full([](const Engine::InfoFull&) { });
    engine.set_on_iter([](const Engine::InfoIter&) { });

    set_option(engine, "RevealBonusBase", 80);
    set_option(engine, "RevealBonusPhase", 24);
    set_option(engine, "RevealBonusPool", -16);
    set_option(engine, "RevealBonusUnknown", 32);
    set_option(engine, "RevealBonusComeback", 64);
    set_option(engine, "RevealMoveOrder", 4096);
    set_option(engine, "RevealReduction", 300);
    set_option(engine, "RevealPruningMargin", 128);
    set_option(engine, "RevealQuietMoveOrder", 2048);

    constexpr auto SingleRevealFEN  = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w P1 0 1";
    constexpr auto MultiRevealFEN   = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w R1P1 0 1";
    constexpr auto NestedRevealFEN  = "4k4/9/9/x8/9/4P4/9/9/9/X3K4 w R1P1r2p3 0 1";
    constexpr auto MultiMoveFEN     = "4k4/9/9/9/9/p8/X1X6/9/9/4K4 w P2 0 1";
    constexpr auto InCheckRevealFEN = "4r3k/9/9/9/9/9/9/9/9/3XK4 w A1 0 1";

    search_once(engine, SingleRevealFEN, 1, "a3a4");  // qsearch -> flip_search
    search_once(engine, MultiRevealFEN, 4, "a3a4");   // multi-identity PV search
    search_once(engine, NestedRevealFEN, 4, "a0a6");  // dark captures dark, two aggregates

    const std::string first = search_once(engine, MultiMoveFEN, 4);
    engine.search_clear();
    require(search_once(engine, MultiMoveFEN, 4) == first,
            "LMR/PV reveal smoke was not deterministic");

    StateInfo inCheckState;
    Position  inCheck;
    inCheck.set(InCheckRevealFEN, &inCheckState);
    const Move block(SQ_D0, SQ_E1);
    require(bool(inCheck.checkers()), "in-check reveal fixture is not in check");
    require(inCheck.move_dark(block) && inCheck.pseudo_legal(block) && inCheck.legal(block),
            "in-check reveal fixture lacks its dark evasion");
    search_once(engine, InCheckRevealFEN, 2, "d0e1");
}

void test_combined_reward_and_safe_reduction_search_routes(Engine& engine) {
    constexpr auto SingleRevealFEN = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w P1 0 1";
    constexpr auto MultiRevealFEN  = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w R1P1 0 1";
    constexpr auto NestedRevealFEN = "4k4/9/9/x8/9/4P4/9/9/9/X3K4 w R1P1r2p3 0 1";
    constexpr auto InCheckRevealFEN = "4r3k/9/9/9/9/9/9/9/9/3XK4 w A1 0 1";
    constexpr auto CounterRevealFEN = "4k4/9/9/9/4P4/9/X8/9/9/4K4 w P1 37 88";
    constexpr auto DarkCaptureFEN = "4k4/9/9/9/4P4/p8/X8/9/9/4K4 w P1 0 1";
    constexpr auto ReductionRouteFEN =
      "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w "
      "R2A2C2P5N2B2r2a2c2p5n2b2 0 1";
    constexpr auto QsearchEvasionFEN =
      "k2r5/9/9/9/9/9/9/9/3P1P3/3XKP3 b A1 0 1";

    set_option(engine, "Threads", 1);
    set_option(engine, "Hash", 1);
    reset_reveal_options(engine);

    const SearchSignature singleBase =
      fresh_search_signature(engine, SingleRevealFEN, 1, "a3a4");
    require(singleBase.bestMove.compare(0, 4, "a3a4") == 0 && singleBase.nodes > 0,
            "single-identity qsearch fixture did not produce a complete forced signature");
    set_option(engine, "RevealQuietBase", 80);
    set_option(engine, "RevealQuietSafety", 96);
    const SearchSignature singlePositive =
      fresh_search_signature(engine, SingleRevealFEN, 1, "a3a4");
    require(singlePositive.bestMove.compare(0, 4, "a3a4") == 0 && singlePositive.nodes > 0,
            "positive single-identity quiet reward lost the forced move or node report");
    require(uci_score_matches_bonus(SingleRevealFEN, singleBase, singlePositive, 176, 1),
            "single-identity quiet reward was not applied exactly once after qsearch/flip aggregate");
    require(!uci_score_matches_bonus(SingleRevealFEN, singleBase, singlePositive, 176, 2),
            "single-identity quiet reward matches a double application");

    reset_reveal_options(engine);
    const SearchSignature multiBase =
      fresh_search_signature(engine, MultiRevealFEN, 4, "a3a4");
    set_option(engine, "RevealQuietBase", -80);
    set_option(engine, "RevealQuietSafety", -96);
    const SearchSignature multiNegative =
      fresh_search_signature(engine, MultiRevealFEN, 4, "a3a4");
    require(multiNegative.bestMove.compare(0, 4, "a3a4") == 0 && multiNegative.nodes > 0,
            "negative multi-identity quiet reward lost the forced PV move or node report");
    require(uci_score_matches_bonus(MultiRevealFEN, multiBase, multiNegative, -176, 1),
            "multi-identity quiet reward was not applied once outside the identity aggregate");
    require(!uci_score_matches_bonus(MultiRevealFEN, multiBase, multiNegative, -176, 2),
            "multi-identity quiet reward matches an identity-loop/double application");

    reset_reveal_options(engine);
    const SearchSignature nestedBase =
      fresh_search_signature(engine, NestedRevealFEN, 4, "a0a6");
    set_option(engine, "RevealQuietBase", 80);
    set_option(engine, "RevealQuietSafety", 96);
    const SearchSignature nestedQuiet =
      fresh_search_signature(engine, NestedRevealFEN, 4, "a0a6");
    require(same_signature(nestedBase, nestedQuiet),
            "V2 quiet reward changed nested dark-captures-dark score/nodes");

    reset_reveal_options(engine);
    const SearchSignature nestedOnceBase =
      fresh_search_signature(engine, NestedRevealFEN, 1, "a0a6");
    set_option(engine, "RevealBonusBase", 80);
    const SearchSignature nestedV1 =
      fresh_search_signature(engine, NestedRevealFEN, 1, "a0a6");
    require(uci_score_matches_bonus(NestedRevealFEN, nestedOnceBase, nestedV1, 80, 1),
            "nested dark-captures-dark V1 reward was not applied once outside both aggregates");
    require(!uci_score_matches_bonus(NestedRevealFEN, nestedOnceBase, nestedV1, 80, 2),
            "nested dark-captures-dark V1 reward matches a hidden/flip double application");

    reset_reveal_options(engine);
    const SearchSignature captureBase =
      fresh_search_signature(engine, DarkCaptureFEN, 1, "a3a4");
    set_option(engine, "RevealQuietBase", 80);
    set_option(engine, "RevealQuietSafety", 96);
    const SearchSignature captureV2 =
      fresh_search_signature(engine, DarkCaptureFEN, 1, "a3a4");
    require(same_signature(captureBase, captureV2),
            "dark capture/ordinary captures-only qsearch received a V2 contribution");
    reset_reveal_options(engine);
    set_option(engine, "RevealBonusBase", 80);
    const SearchSignature captureV1 =
      fresh_search_signature(engine, DarkCaptureFEN, 1, "a3a4");
    require(uci_score_matches_bonus(DarkCaptureFEN, captureBase, captureV1, 80, 1),
            "dark capture stopped receiving its V1 reward");
    require(!uci_score_matches_bonus(DarkCaptureFEN, captureBase, captureV1, 80, 2),
            "dark capture V1 reward matches a hidden/flip double application");

    reset_reveal_options(engine);
    const SearchSignature evasionBase =
      fresh_search_signature(engine, InCheckRevealFEN, 2, "d0e1");
    set_option(engine, "RevealQuietBase", 80);
    set_option(engine, "RevealQuietSafety", 96);
    const SearchSignature evasionPositive =
      fresh_search_signature(engine, InCheckRevealFEN, 2, "d0e1");
    require(evasionPositive.bestMove.compare(0, 4, "d0e1") == 0 && evasionPositive.nodes > 0,
            "in-check dark quiet evasion did not produce a complete signature");
    require(!same_signature(evasionBase, evasionPositive),
            "in-check qsearch/PV quiet evasion ignored V2 reward");

    StateInfo qsearchState;
    Position  qsearchPosition;
    qsearchPosition.set(QsearchEvasionFEN, &qsearchState);
    const Move checkingMove = UCIEngine::to_move(qsearchPosition, "d9e9");
    require(checkingMove.is_ok() && qsearchPosition.legal(checkingMove)
              && !qsearchPosition.move_dark(checkingMove)
              && qsearchPosition.gives_check(checkingMove),
            "qsearch evasion fixture lacks its visible checking move");
    StateInfo afterCheck;
    qsearchPosition.do_move(checkingMove, afterCheck);
    const Move quietEvasion = UCIEngine::to_move(qsearchPosition, "d0e1");
    require(bool(qsearchPosition.checkers()) && quietEvasion.is_ok()
              && qsearchPosition.legal(quietEvasion) && qsearchPosition.move_dark(quietEvasion)
              && !qsearchPosition.capture(quietEvasion),
            "qsearch evasion fixture lacks its legal dark quiet response");
    require(Reveal::quiet_move_context(qsearchPosition, quietEvasion, true).eligible,
            "qsearch dark quiet response is not V2 eligible");
    qsearchPosition.undo_move(checkingMove);

    reset_reveal_options(engine);
    const SearchSignature qsearchBase =
      fresh_search_signature(engine, QsearchEvasionFEN, 1, "d9e9");
    set_option(engine, "RevealQuietBase", 80);
    const SearchSignature qsearchPositive =
      fresh_search_signature(engine, QsearchEvasionFEN, 1, "d9e9");
    require(uci_score_matches_bonus(QsearchEvasionFEN, qsearchBase, qsearchPositive, -80, 1),
            "in-check qsearch quiet evasion did not apply positive V2 once for the child side");
    set_option(engine, "RevealQuietBase", -80);
    const SearchSignature qsearchNegative =
      fresh_search_signature(engine, QsearchEvasionFEN, 1, "d9e9");
    require(uci_score_matches_bonus(QsearchEvasionFEN, qsearchBase, qsearchNegative, 80, 1),
            "in-check qsearch quiet evasion did not apply negative V2 once for the child side");

    reset_reveal_options(engine);
    set_option(engine, "RevealQuietReduction", 545);
    const SearchSignature baseCounters =
      fresh_search_signature(engine, SingleRevealFEN, 4, "a3a4");
    const SearchSignature changedCounters =
      fresh_search_signature(engine, CounterRevealFEN, 4, "a3a4");
    require(baseCounters.bestMove == changedCounters.bestMove,
            "RevealQuietReduction search decision depends on FEN counters");

    reset_reveal_options(engine);
    set_option(engine, "RevealQuietReduction", 545);
    const SearchSignature reductionPositive =
      fresh_search_signature(engine, ReductionRouteFEN, 4);
    set_option(engine, "RevealQuietReduction", -545);
    const SearchSignature reductionNegative =
      fresh_search_signature(engine, ReductionRouteFEN, 4);
    require(reductionPositive.nodes == 346 && reductionNegative.nodes == 346,
            "integrated active-flip depth-4 route signature changed: positive="
              + std::to_string(reductionPositive.nodes)
              + " negative=" + std::to_string(reductionNegative.nodes));

    reset_reveal_options(engine);
    set_option(engine, "RevealQuietBase", 80);
    set_option(engine, "RevealQuietSafety", 96);
    set_option(engine, "RevealQuietMoveOrder", 1024);
    set_option(engine, "RevealQuietReduction", 545);
    const SearchSignature fullPositive =
      fresh_search_signature(engine, MultiRevealFEN, 4, "a3a4");
    set_option(engine, "RevealQuietBase", -80);
    set_option(engine, "RevealQuietSafety", -96);
    set_option(engine, "RevealQuietMoveOrder", -1024);
    set_option(engine, "RevealQuietReduction", -545);
    const SearchSignature fullNegative =
      fresh_search_signature(engine, MultiRevealFEN, 4, "a3a4");
    require(fullPositive.bestMove.compare(0, 4, "a3a4") == 0
              && fullPositive.score == "cp 237" && fullPositive.nodes == 17
              && fullNegative.bestMove.compare(0, 4, "a3a4") == 0
              && fullNegative.score == "cp 198" && fullNegative.nodes == 17,
            "full V2 PV/nonPV/LMR signature changed: positive=" + fullPositive.bestMove + "/"
              + fullPositive.score + "/" + std::to_string(fullPositive.nodes) + " negative="
              + fullNegative.bestMove + "/" + fullNegative.score + "/"
              + std::to_string(fullNegative.nodes));

    reset_reveal_options(engine);
    set_option(engine, "RevealQuietBase", 80);
    set_option(engine, "RevealQuietSafety", 96);
    set_option(engine, "RevealQuietMoveOrder", 1024);
    set_option(engine, "RevealQuietReduction", 545);
    const SearchSignature lmrPositive =
      fresh_search_signature(engine, ReductionRouteFEN, 4);
    set_option(engine, "RevealQuietBase", -80);
    set_option(engine, "RevealQuietSafety", -96);
    set_option(engine, "RevealQuietMoveOrder", -1024);
    set_option(engine, "RevealQuietReduction", -545);
    const SearchSignature lmrNegative =
      fresh_search_signature(engine, ReductionRouteFEN, 4);
    require(lmrPositive.bestMove == "a3a4" && lmrPositive.score == "cp 34"
              && lmrPositive.nodes == 373 && lmrNegative.bestMove == "a3a4"
              && lmrNegative.score == "cp -3" && lmrNegative.nodes == 417,
            "full-board V2 LMR signature changed: positive=" + lmrPositive.bestMove + "/"
              + lmrPositive.score + "/" + std::to_string(lmrPositive.nodes) + " negative="
              + lmrNegative.bestMove + "/" + lmrNegative.score + "/"
              + std::to_string(lmrNegative.nodes));

    reset_reveal_options(engine);
}

}  // namespace

int main() {
    try
    {
        Bitboards::init();
        Position::init();
        Engine engine;
        // Runtime no longer selects an NNUE file implicitly.  The test
        // fixture is supplied explicitly through the same load path used by
        // the UCI EvalFile option.
        const char* package = std::getenv("ABJCHESS_V11_TEST_PACKAGE");
        engine.load_big_network(package ? package : "abjchess-v11.nnue");
        engine.verify_networks();
        test_engine_banner_is_utf8();
        test_round_half_away_from_zero();
        test_integrated_search_contracts();
        test_feature_endpoints_and_pool_counts();
        test_quiet_features();
        test_quiet_safety();
        test_bonus_rounding_clamping_and_disabled_fast_path();
        test_quiet_bonus_and_combination();
        test_combined_bonus_and_lazy_raw_eval_gates_follow_move_context();
        test_position_features_are_side_to_move_symmetric();
        test_quiet_position_features_use_side_pool_and_ignore_counters();
        test_reward_transform_and_generalized_inverse();
        test_wrapper_applies_bonus_once_to_aggregated_result();
        test_pruning_and_move_order_helpers_preserve_zero_path();
        test_v2_ordering_helpers_are_selective_and_reversible();
        test_quiet_move_context_is_selective_safe_and_counter_independent();
        test_reveal_reduction_combines_v1_and_safe_only_v2_without_ply_scaling();
        test_move_picker_orders_only_safe_dark_quiets_and_preserves_tt_priority();
        test_move_picker_keeps_v2_off_captures_and_v1_on_dark_captures();
        test_good_quiet_bucket_uses_raw_score_after_safe_ordering();
        test_good_capture_see_uses_raw_score_after_dark_ordering();
        test_probcut_qualification_uses_only_see_and_pruning_margin();
        test_quiet_pruning_limits_apply_after_the_current_move();
        test_research_raw_eval_uses_current_uncorrected_tt_value_only();
        test_reveal_uci_defaults_and_ranges(engine);
        test_reveal_worker_parameter_snapshot(engine);
        test_reveal_quiet_options_clear_search(engine);
        test_reveal_search_paths_and_in_check_comeback(engine);
#if !defined(ABJCHESS_V11)
        // Exact score/node snapshots depend on the selected NNUE package;
        // structural reveal-path coverage is the portable contract.
        test_combined_reward_and_safe_reduction_search_routes(engine);
#endif
        std::cout << "reveal tests passed\n";
        return 0;
    } catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
