#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "abjnnue/abjnnue_accumulator.h"
#include "abjnnue/abjnnue_feature_encoder.h"
#include "abjnnue/abjnnue_inference.h"
#include "abjnnue/abjnnue_layers.h"
#include "abjnnue/abjnnue_model.h"
#include "bitboard.h"
#include "engine.h"
#include "position.h"
#include "search.h"
#include "types.h"

namespace Stockfish::Search {

int hidden_capture_candidates(const Position& position,
                              Color           rootObserver,
                              bool            capturedDark,
                              Position::RestPieceList& candidates);

}

namespace {

using namespace Stockfish;

constexpr auto VisibleStartFEN =
  "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - 0 1";
constexpr auto HiddenCaptureFEN = "4k4/9/9/9/9/p8/X8/9/9/4K4 w P1 0 1";
constexpr auto MultiHiddenCaptureFEN = "4k4/9/9/9/9/p8/X1X6/9/9/4K4 w P2 0 1";
constexpr auto VisibleCapturesDarkFEN =
  "4k4/9/9/x8/R8/4P4/9/9/9/4K4 w r2p3 0 1";
constexpr auto DarkCapturesDarkFEN =
  "4k4/9/9/x8/9/4P4/9/9/9/X3K4 w R1r2p3 0 1";
// The black rook captures White's only visible rook.  This changes White's
// 4-way attack bucket from rook+minor to minor-free without changing the
// ordinary board feature delta shape.
constexpr auto AttackBucketTransitionFEN =
  "r3k4/9/9/9/4P4/9/9/9/R8/4K4 b - 0 1";
// Moving a visible rook across the river's file mirror changes the canonical
// MidMirrorEncoding decision while retaining the same piece/attack counts.
constexpr auto MidMirrorTransitionFEN =
  "4k4/9/9/9/4P4/9/9/9/R8/4K4 w - 0 1";

struct JsonValue {
    enum class Type { Integer, String, Array, Object };

    Type                         type = Type::Integer;
    std::int64_t                 integer = 0;
    std::string                  string;
    std::vector<JsonValue>       array;
    std::map<std::string, JsonValue> object;

    const JsonValue& at(std::string_view key) const {
        if (type != Type::Object) throw std::runtime_error("golden JSON value is not an object");
        const auto found = object.find(std::string(key));
        if (found == object.end())
            throw std::runtime_error("golden JSON object is missing field: " + std::string(key));
        return found->second;
    }

    const JsonValue* find(std::string_view key) const {
        if (type != Type::Object) return nullptr;
        const auto found = object.find(std::string(key));
        return found == object.end() ? nullptr : &found->second;
    }

    std::int64_t as_integer() const {
        if (type != Type::Integer) throw std::runtime_error("golden JSON value is not an integer");
        return integer;
    }

    const std::string& as_string() const {
        if (type != Type::String) throw std::runtime_error("golden JSON value is not a string");
        return string;
    }

    const std::vector<JsonValue>& as_array() const {
        if (type != Type::Array) throw std::runtime_error("golden JSON value is not an array");
        return array;
    }
};

class JsonParser {
   public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    JsonValue parse() {
        JsonValue value = parse_value();
        skip_space();
        if (position_ != input_.size()) throw std::runtime_error("trailing golden JSON data");
        return value;
    }

   private:
    void skip_space() {
        while (position_ < input_.size()
               && (input_[position_] == ' ' || input_[position_] == '\n'
                   || input_[position_] == '\r' || input_[position_] == '\t'))
            ++position_;
    }

    char take() {
        if (position_ >= input_.size()) throw std::runtime_error("truncated golden JSON");
        return input_[position_++];
    }

    void expect(char expected) {
        skip_space();
        if (take() != expected) throw std::runtime_error("invalid golden JSON punctuation");
    }

    std::string parse_string() {
        skip_space();
        if (take() != '"') throw std::runtime_error("golden JSON string is missing quote");
        std::string value;
        for (;;)
        {
            const char token = take();
            if (token == '"') return value;
            if (static_cast<unsigned char>(token) < 0x20)
                throw std::runtime_error("golden JSON string contains a control character");
            if (token != '\\')
            {
                value.push_back(token);
                continue;
            }
            switch (take())
            {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            default: throw std::runtime_error("unsupported golden JSON string escape");
            }
        }
    }

    JsonValue parse_integer() {
        skip_space();
        const std::size_t begin = position_;
        if (position_ < input_.size() && input_[position_] == '-') ++position_;
        while (position_ < input_.size() && input_[position_] >= '0'
               && input_[position_] <= '9')
            ++position_;
        if (position_ == begin || (position_ == begin + 1 && input_[begin] == '-'))
            throw std::runtime_error("invalid golden JSON integer");
        std::int64_t value = 0;
        const auto result = std::from_chars(input_.data() + begin,
                                            input_.data() + position_, value);
        if (result.ec != std::errc{} || result.ptr != input_.data() + position_)
            throw std::runtime_error("golden JSON integer is outside int64");
        JsonValue output;
        output.type = JsonValue::Type::Integer;
        output.integer = value;
        return output;
    }

    JsonValue parse_array() {
        JsonValue output;
        output.type = JsonValue::Type::Array;
        expect('[');
        skip_space();
        if (position_ < input_.size() && input_[position_] == ']')
        {
            ++position_;
            return output;
        }
        for (;;)
        {
            output.array.push_back(parse_value());
            skip_space();
            const char separator = take();
            if (separator == ']') return output;
            if (separator != ',') throw std::runtime_error("invalid golden JSON array");
        }
    }

    JsonValue parse_object() {
        JsonValue output;
        output.type = JsonValue::Type::Object;
        expect('{');
        skip_space();
        if (position_ < input_.size() && input_[position_] == '}')
        {
            ++position_;
            return output;
        }
        for (;;)
        {
            std::string key = parse_string();
            expect(':');
            if (!output.object.emplace(std::move(key), parse_value()).second)
                throw std::runtime_error("duplicate golden JSON object key");
            skip_space();
            const char separator = take();
            if (separator == '}') return output;
            if (separator != ',') throw std::runtime_error("invalid golden JSON object");
        }
    }

    JsonValue parse_value() {
        skip_space();
        if (position_ >= input_.size()) throw std::runtime_error("truncated golden JSON value");
        if (input_[position_] == '{') return parse_object();
        if (input_[position_] == '[') return parse_array();
        if (input_[position_] == '"')
        {
            JsonValue output;
            output.type = JsonValue::Type::String;
            output.string = parse_string();
            return output;
        }
        return parse_integer();
    }

    std::string_view input_;
    std::size_t      position_ = 0;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::int64_t json_integer(const JsonValue& value, std::string_view label) {
    try
    {
        return value.as_integer();
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(std::string(label) + ": " + error.what());
    }
}

std::string fixture_fen(const JsonValue& testCase) {
    static constexpr char RestPieces[2][6] = {
      {'R', 'A', 'C', 'P', 'N', 'B'},
      {'r', 'a', 'c', 'p', 'n', 'b'}};
    const auto& sides = testCase.at("rest").as_array();
    require(sides.size() == 2, "golden fixture rest must have two sides");
    std::string rest;
    for (std::size_t side = 0; side < sides.size(); ++side)
    {
        const auto& counts = sides[side].as_array();
        require(counts.size() == 6, "golden fixture rest side must have six counts");
        for (std::size_t type = 0; type < counts.size(); ++type)
        {
            const auto count = json_integer(counts[type], "golden fixture rest count");
            require(count >= 0 && count <= 5, "golden fixture rest count is out of range");
            if (count != 0)
            {
                rest.push_back(RestPieces[side][type]);
                rest += std::to_string(count);
            }
        }
    }
    if (rest.empty()) rest = "-";
    const JsonValue* sideToMove = testCase.find("side_to_move");
    const auto side = sideToMove ? json_integer(*sideToMove, "golden side_to_move") : 0;
    require(side == 0 || side == 1, "golden side_to_move is outside 0..1");
    return testCase.at("board").as_string() + (side == 0 ? " w " : " b ")
         + rest + " 0 1";
}

std::array<std::uint8_t, 9> fixture_offsets(const JsonValue& perspective) {
    const auto& values = perspective.at("meta_feature_offsets").as_array();
    require(values.size() == 9, "golden metadata offset list must contain nine entries");
    std::array<std::uint8_t, 9> result{};
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const auto value = json_integer(values[i], "golden metadata offset");
        require(value >= 0 && value < 100, "golden metadata offset is outside 0..99");
        result[i] = static_cast<std::uint8_t>(value);
    }
    return result;
}

void test_v11_feature_golden(const std::filesystem::path& fixture) {
    std::ifstream stream(fixture, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open V11 feature golden fixture");
    const std::string json((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    const JsonValue root = JsonParser(json).parse();
    require(root.at("schema").as_string() == "abjchess-v11-feature-golden-v1",
            "V11 feature golden schema differs");
    require(root.at("feature_identity_sha256").as_string()
              == "ecdc69d39f113c79f75ce1932f176f38eba5f0f2f220bb89745709760f934cfa",
            "V11 feature golden identity differs");

    const auto& cases = root.at("cases").as_array();
    require(cases.size() >= 8, "V11 feature golden fixture is incomplete");
    for (const JsonValue& testCase : cases)
    {
        const std::string label = testCase.at("name").as_string();
        std::deque<StateInfo> states(1);
        Position position;
        try
        {
            position.set(fixture_fen(testCase), &states.back());
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error(label + ": cannot construct golden position: " + error.what());
        }
        const auto encoded = ABJNNUE::FeatureEncoder::encode(position);
        const auto& expectedSelection = testCase.at("selection");
        require(encoded.layerStackSelection.floor
                  == json_integer(expectedSelection.at("floor"), "golden floor"),
                label + ": layer-stack floor differs");
        require(encoded.layerStackSelection.blendQ8
                  == json_integer(expectedSelection.at("blend_q8"), "golden blend_q8"),
                label + ": layer-stack blend differs");

        const auto& perspectives = testCase.at("perspectives");
        for (Color perspective : {WHITE, BLACK})
        {
            const auto& expected = perspectives.at(perspective == WHITE ? "0" : "1");
            const auto offsets = fixture_offsets(expected);
            require(encoded.perspectives[perspective].metaOffsets == offsets,
                    label + ": metadata offsets differ for perspective "
                      + std::to_string(int(perspective)));
            require(offsets[5] - 40
                      == json_integer(expected.at("unknown_loss_count"),
                                      "golden unknown-loss count"),
                    label + ": own unknown-loss bucket differs");
            require(offsets[6] - 56
                      == json_integer(expected.at("enemy_unknown_loss"),
                                      "golden enemy unknown-loss count"),
                    label + ": enemy unknown-loss count differs");
            const auto& threats = expected.at("threat_summary").as_array();
            require(threats.size() == 2, "golden threat summary must contain two counts");
            require(offsets[7] - 72 == json_integer(threats[0], "golden enemy threat")
                      && offsets[8] - 80 == json_integer(threats[1], "golden own threat"),
                    label + ": threat summary differs");
        }
    }
}

template<typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try
    {
        function();
    }
    catch (const Exception&)
    {
        return;
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(message + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(message + ": no exception");
}

void compare_accumulated(const ABJNNUE::AccumulatedPosition& actual,
                         const ABJNNUE::AccumulatedPosition& expected,
                         const std::string& label) {
    for (Color perspective : {WHITE, BLACK})
        for (std::size_t i = 0; i < ABJNNUE::RuntimeLayout::AccumulatorWidth; ++i)
            if (actual.perspectives[perspective].values[i]
                != expected.perspectives[perspective].values[i])
                throw std::runtime_error(
                  label + ": accumulator differs for "
                  + (perspective == WHITE ? "white" : "black") + " at index "
                  + std::to_string(i) + ": actual="
                  + std::to_string(actual.perspectives[perspective].values[i])
                  + " expected="
                  + std::to_string(expected.perspectives[perspective].values[i]));
}

void compare_stack_with_full(const ABJNNUE::Model& model,
                             const Position& position,
                             ABJNNUE::AccumulatorStack& stack,
                             const std::string& label) {
    const auto full = ABJNNUE::Inference::evaluate(model, position);
    const auto view = stack.evaluate(model, position);
    compare_accumulated(view.accumulated, full.accumulated, label);
    require(view.layerStackBucket == full.encoded.layerStackSelection.floor,
            label + ": layer-stack floor differs");
    require(view.layerStackBlendQ8 == full.encoded.layerStackSelection.blendQ8,
            label + ": layer-stack blend differs");
    const auto incremental = ABJNNUE::Inference::evaluate_accumulated(
      model, position, view.accumulated,
      ABJNNUE::LayerStackSelection{view.layerStackBucket, view.layerStackBlendQ8});
    require(incremental.positionalRaw == full.positionalRaw,
            label + ": positional inference differs");
}

void test_runtime_optimization_contract(const ABJNNUE::Model& model) {
    static_assert(ABJNNUE::FeatureEncoder::MaxActiveFeatures == 160,
                  "the V11 feature list must cover board, meta, and rest features");

    std::deque<StateInfo> states(1);
    Position position;
    position.set(VisibleStartFEN, &states.back());
    auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
    auto cache = std::make_unique<ABJNNUE::RefreshCache>();

    const auto first = stack->evaluate(model, position, *cache);
    (void) first;
#if defined(ABJNNUE_RUNTIME_STATS)
    require(cache->stats().lookups != 0, "refresh cache was not consulted");
#endif

    stack->reset();
    const auto cachedRoot = stack->evaluate(model, position, *cache);
    const auto fullRoot  = ABJNNUE::Inference::evaluate(model, position);
    compare_accumulated(cachedRoot.accumulated, fullRoot.accumulated, "cached root");
    require(cachedRoot.layerStackBucket == fullRoot.encoded.layerStackSelection.floor
              && cachedRoot.layerStackBlendQ8 == fullRoot.encoded.layerStackSelection.blendQ8,
            "cached root did not retain layer-stack selection");
#if defined(ABJNNUE_RUNTIME_STATS)
    require(cache->stats().hits != 0, "refresh cache did not hit on a repeated position");
#endif

    const Move move(SQ_A3, SQ_A4);
    states.emplace_back();
    const auto dirty = position.do_move(move, states.back(), position.gives_check(move), nullptr);
    stack->push(dirty, position);
    const auto second = stack->evaluate(model, position, *cache);
    (void) second;
#if defined(ABJNNUE_RUNTIME_STATS) && defined(ABJNNUE_RUNTIME_FUSED_UPDATE)
        require(stack->stats().fusedUpdates != 0, "fused accumulator update was not used");
#endif

    stack->reset();
    const auto cachedMove = stack->evaluate(model, position, *cache);
    const auto fullMove   = ABJNNUE::Inference::evaluate(model, position);
    compare_accumulated(cachedMove.accumulated, fullMove.accumulated, "cached move");
    require(cachedMove.layerStackBucket == fullMove.encoded.layerStackSelection.floor
              && cachedMove.layerStackBlendQ8 == fullMove.encoded.layerStackSelection.blendQ8,
            "cached move did not retain layer-stack selection");
#if defined(ABJNNUE_RUNTIME_STATS)
    require(cache->stats().deltaRows != 0, "refresh cache did not apply feature deltas");
#endif

    ABJNNUE::TransformedFeatures transformed{};
    const auto raw = ABJNNUE::Inference::evaluate_accumulated(
      model, position, second.accumulated,
      ABJNNUE::LayerStackSelection{second.layerStackBucket, second.layerStackBlendQ8},
      &transformed);
    (void) raw;
#if defined(ABJNNUE_RUNTIME_STATS) && defined(ABJNNUE_RUNTIME_SIMD_TRANSFORM)
    require(ABJNNUE::Inference::stats().simdTransforms != 0,
            "SIMD pairwise transform was not used");
#endif
}

std::int64_t interpolate_q8_oracle(std::int64_t first,
                                   std::int64_t second,
                                   std::uint8_t blendQ8) {
    if (blendQ8 == 0) return first;
    if (blendQ8 == 255) return second;
    const std::int64_t q = blendQ8;
    return (first * (255 - q) + second * q + 127) / 255;
}

void test_interpolation_oracle(const ABJNNUE::Model& model) {
    require(ABJNNUE::RuntimeLayout::interpolate_q8(-102, -104, 0) == -102,
            "signed interpolation changed the q=0 endpoint");
    require(ABJNNUE::RuntimeLayout::interpolate_q8(-102, -104, 128) == -102,
            "signed interpolation midpoint differs from the int64 oracle");
    require(ABJNNUE::RuntimeLayout::interpolate_q8(-102, -104, 255) == -104,
            "signed interpolation changed the q=255 endpoint");

    std::deque<StateInfo> states(1);
    Position position;
    position.set(VisibleStartFEN, &states.back());

    ABJNNUE::AccumulatedPosition accumulated{};
    constexpr std::uint32_t floor = 8;
    const Color us = position.side_to_move();
    const auto context = ABJNNUE::FeatureEncoder::inventory_context(position, us);

    ABJNNUE::TransformedFeatures transformed{};
    for (std::size_t i = 0; i < transformed.size(); ++i)
        transformed[i] = static_cast<std::uint8_t>((i * 37 + 11) % 127);
    const auto firstHead = static_cast<std::int64_t>(model.propagate(floor, transformed.data(), context));
    const auto secondHead = static_cast<std::int64_t>(
      model.propagate(floor + 1, transformed.data(), context));

    for (const std::uint8_t q : {std::uint8_t(0), std::uint8_t(128), std::uint8_t(255)})
    {
        const auto expectedHead = interpolate_q8_oracle(firstHead, secondHead, q);
        require(model.propagate_interpolated(floor, q, transformed.data(), context) == expectedHead,
                "dense interpolation differs from signed-int64 oracle at q="
                  + std::to_string(q));

        for (Color perspective : {WHITE, BLACK})
            for (std::size_t i = 0; i < ABJNNUE::RuntimeLayout::PerspectiveOutputWidth; ++i)
            {
                const std::size_t output = (perspective == us ? 0 : 1)
                                         * ABJNNUE::RuntimeLayout::PerspectiveOutputWidth + i;
                const std::uint8_t value = transformed[output];
                accumulated.perspectives[perspective].values[i] = value == 0 ? 0 : 127;
                accumulated.perspectives[perspective].values[
                  i + ABJNNUE::RuntimeLayout::PerspectiveOutputWidth] =
                  value == 0 ? 0 : static_cast<std::int16_t>(value + 1);
            }
        ABJNNUE::TransformedFeatures actualTransformed{};
        const auto raw = ABJNNUE::Inference::evaluate_accumulated(
          model, position, accumulated, ABJNNUE::LayerStackSelection{floor, q},
          &actualTransformed);
        require(actualTransformed == transformed,
                "interpolation oracle fixture did not reproduce transformed input");
        require(raw.positionalRaw == expectedHead,
                "inference dense selection differs from model interpolation at q="
                  + std::to_string(q));

    }
}

std::array<std::uint8_t, ABJNNUE::RuntimeLayout::HeadInputWidth> make_head_input(
  const ABJNNUE::Model& model,
  const ABJNNUE::TransformedFeatures& transformed,
  const ABJNNUE::InventoryContext& context) {
    std::array<std::uint8_t, ABJNNUE::RuntimeLayout::HeadInputWidth> input{};
    std::copy(transformed.begin(), transformed.end(), input.begin());
    for (std::size_t output = 0; output < ABJNNUE::RuntimeLayout::InventoryContextHidden; ++output)
    {
        std::int64_t sum = model.context_biases()[output];
        for (std::size_t feature = 0; feature < ABJNNUE::RuntimeLayout::InventoryContextInputs; ++feature)
            sum += std::int32_t(context[feature])
                 * std::int32_t(model.context_weights()[output
                     * ABJNNUE::RuntimeLayout::InventoryContextInputs + feature]);
        input[ABJNNUE::RuntimeLayout::AccumulatorWidth + output] =
          static_cast<std::uint8_t>(std::clamp<std::int64_t>(sum >> 7, 0, 127));
    }
    return input;
}

void test_pairwise_transform_boundaries(const ABJNNUE::Model& model) {
    std::deque<StateInfo> states(1);
    Position position;
    position.set(VisibleStartFEN, &states.back());

    ABJNNUE::AccumulatedPosition accumulated{};
    std::uint32_t random = 0x13579bdfU;
    for (Color perspective : {WHITE, BLACK})
        for (std::size_t i = 0; i < ABJNNUE::RuntimeLayout::AccumulatorWidth; ++i)
        {
            random = random * 1664525U + 1013904223U;
            switch (i % 8)
            {
            case 0: accumulated.perspectives[perspective].values[i] = -32768; break;
            case 1: accumulated.perspectives[perspective].values[i] = -1024; break;
            case 2: accumulated.perspectives[perspective].values[i] = -1; break;
            case 3: accumulated.perspectives[perspective].values[i] = 0; break;
            case 4: accumulated.perspectives[perspective].values[i] = 1; break;
            case 5: accumulated.perspectives[perspective].values[i] = 127; break;
            case 6: accumulated.perspectives[perspective].values[i] = 32767; break;
            default:
                accumulated.perspectives[perspective].values[i] =
                  static_cast<std::int16_t>(random & 0xffffU);
                break;
            }
        }

    ABJNNUE::TransformedFeatures actual{};
    (void) ABJNNUE::Inference::evaluate_accumulated(model, position, accumulated, 0, &actual);
    const Color order[2] = {position.side_to_move(), ~position.side_to_move()};
    for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t i = 0; i < ABJNNUE::RuntimeLayout::PerspectiveOutputWidth; ++i)
        {
            const auto& values = accumulated.perspectives[order[p]].values;
            const int   a = std::clamp<int>(values[i], 0, 127);
            const int   b = std::clamp<int>(
              values[i + ABJNNUE::RuntimeLayout::PerspectiveOutputWidth], 0, 127);
            const auto expected = static_cast<std::uint8_t>((a * b) >> 7);
            require(actual[p * ABJNNUE::RuntimeLayout::PerspectiveOutputWidth + i] == expected,
                    "pairwise transform differs at boundary sample");
        }
}

void test_refresh_cache_hidden_and_fallback(const ABJNNUE::Model& model) {
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set(MultiHiddenCaptureFEN, &states.back());
        auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
        auto cache = std::make_unique<ABJNNUE::RefreshCache>();

        const auto root = stack->evaluate(model, position, *cache);
        (void) root;
        stack->reset();
        const auto cachedRoot = stack->evaluate(model, position, *cache);
        compare_accumulated(cachedRoot.accumulated,
                            ABJNNUE::Inference::evaluate(model, position).accumulated,
                            "hidden cached root");

        const Move move(SQ_A3, SQ_A4);
        states.emplace_back();
        const auto dirty = position.do_move(move, states.back(), false, nullptr);
        stack->push(dirty, position);
        const auto moved = stack->evaluate(model, position, *cache);
        compare_accumulated(moved.accumulated,
                            ABJNNUE::Inference::evaluate(model, position).accumulated,
                            "hidden cached move");

        auto revealDirty = stack->latest_dirty_piece();
        stack->pop();
        const Piece hiddenPiece = position.do_flip(SQ_A4, W_PAWN, &revealDirty, nullptr);
        stack->push(revealDirty, position);
        const auto revealed = stack->evaluate(model, position, *cache);
        compare_accumulated(revealed.accumulated,
                            ABJNNUE::Inference::evaluate(model, position).accumulated,
                            "hidden cached reveal");
        position.undo_flip(SQ_A4, hiddenPiece);
    }

    // Keep the king transforms and dark variant unchanged while replacing more
    // than the cache delta budget; this must take the full-refresh fallback.
    constexpr auto ManyFeatureChangesFEN =
      "nrbakabrn/9/2c3c2/p1p1p1p1p/9/9/P1P1P1P1P/2C3C2/9/NRBAKABRN w - 0 1";
    std::deque<StateInfo> states(1);
    Position position;
    position.set(VisibleStartFEN, &states.back());
    auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
    auto cache = std::make_unique<ABJNNUE::RefreshCache>();
    (void) stack->evaluate(model, position, *cache);
    stack->reset();
    states.emplace_back();
    position.set(ManyFeatureChangesFEN, &states.back());
    const auto changed = stack->evaluate(model, position, *cache);
    compare_accumulated(changed.accumulated,
                        ABJNNUE::Inference::evaluate(model, position).accumulated,
                        "cache fallback");
#if defined(ABJNNUE_RUNTIME_STATS)
    require(cache->stats().fallbacks != 0, "cache did not record a large-delta fallback");
#endif
}

void test_heads(const ABJNNUE::Model& model, bool expectHeadSnapshot) {
    ABJNNUE::TransformedFeatures transformed{};
    for (std::size_t i = 0; i < transformed.size(); ++i)
        transformed[i] = static_cast<std::uint8_t>((i * 37 + 11) & 127);
    const ABJNNUE::InventoryContext context{};

    std::array<std::int32_t, ABJNNUE::RuntimeLayout::LayerStacks> actual{};
    for (std::size_t bucket = 0; bucket < actual.size(); ++bucket)
        actual[bucket] = model.propagate(static_cast<std::uint32_t>(bucket), transformed.data(), context);

    if (!expectHeadSnapshot)
        return;

    constexpr std::array<std::int32_t, ABJNNUE::RuntimeLayout::LayerStacks> Expected = {
      260, -1093, -269, -358, -274, -200, -100, -288,
      -659, 1114, -1269, -1639, -34, 590, -279, -588};
    if (actual != Expected)
    {
        std::cerr << "head snapshot:";
        for (const auto value : actual) std::cerr << ' ' << value;
        std::cerr << '\n';
        throw std::runtime_error("16-head runtime snapshot differs");
    }
}

void test_head_backend_equivalence(const ABJNNUE::Model& model) {
#if defined(USE_AVX2)
    require(std::string_view(ABJNNUE::Layers::backend_name()) == "avx2",
            "AVX2 build did not select the AVX2 head backend");
#elif defined(USE_SSSE3)
    require(std::string_view(ABJNNUE::Layers::backend_name()) == "ssse3",
            "SSSE3 build did not select the SSSE3 head backend");
#else
    require(std::string_view(ABJNNUE::Layers::backend_name()) == "scalar",
            "portable build did not select the scalar head backend");
#endif

    ABJNNUE::TransformedFeatures transformed{};
    const ABJNNUE::InventoryContext context{};
    std::uint32_t random = 0x9e3779b9U;
    for (std::size_t sample = 0; sample < 12; ++sample)
    {
        for (std::size_t i = 0; i < transformed.size(); ++i)
        {
            random = random * 1664525U + 1013904223U;
            transformed[i] = sample == 0 ? 0
                           : sample == 1 ? 127
                           : ((sample & 1) && i % 5 != 0)
                           ? 0
                           : static_cast<std::uint8_t>((random >> 25) & 127);
        }
        const auto input = make_head_input(model, transformed, context);
        for (std::size_t bucket = 0; bucket < ABJNNUE::RuntimeLayout::LayerStacks; ++bucket)
        {
            const auto* head = model.eval_heads().data()
                             + bucket * ABJNNUE::RuntimeLayout::EvalHeadBucketSize;
            const auto selected = model.propagate(static_cast<std::uint32_t>(bucket),
                                                  transformed.data(), context);
            const auto scalar = ABJNNUE::Layers::propagate_scalar(head, input.data());
            require(selected == scalar,
                    "runtime head differs from scalar oracle at sample "
                      + std::to_string(sample) + " bucket " + std::to_string(bucket));
        }
    }
}

void test_probability(const ABJNNUE::Model& model, bool expectProbabilitySnapshot) {
    require(model.probability_score_to_mass().size() == 4001,
            "score-to-mass table length differs");
    require(model.probability_mass_to_score().size() == 1901,
            "mass-to-score table length differs");

    const std::vector<std::pair<int, int>> weighted = {{-1500, 2}, {700, 3}, {2000, 1}};
    const auto level3 = model.aggregate_probability(weighted, 3);
    const auto level10 = model.aggregate_probability(weighted, 10);
    const auto positive = model.aggregate_probability({{1500, 1}}, 3);
    const auto negative = model.aggregate_probability({{-1500, 1}}, 3);
    if (expectProbabilitySnapshot
        && (level3 != 162 || level10 != 86 || positive != 1500 || negative != -1500))
    {
        std::cerr << "probability snapshot: " << level3 << ' ' << level10 << ' '
                  << positive << ' ' << negative << '\n';
        throw std::runtime_error("probability runtime snapshot differs");
    }

    require(positive == -negative, "single-score probability aggregation is not exact");
    require(model.aggregate_probability({{11000, 2}, {12000, 1}}, 3) == 11000,
            "positive decisive aggregation differs");
    require(model.aggregate_probability({{-11000, 1}, {-12000, 1}}, 3) == -11000,
            "negative decisive aggregation differs");
    require(model.aggregate_probability({{2100, 1}, {2200, 1}}, 3) == 2150,
            "positive out-of-range aggregation did not preserve the weighted mean");
    require(model.aggregate_probability({{-2100, 1}, {-2200, 1}}, 3) == -2150,
            "negative out-of-range aggregation did not preserve the weighted mean");

    // Level 10 must weight both the mean and variance by each branch count.
    require(model.aggregate_probability({{-2000, 1}, {2000, 31}}, 10) == 1752,
            "level-10 variance ignored branch counts");
    require(model.aggregate_probability({{std::numeric_limits<int>::min(), 1},
                                         {std::numeric_limits<int>::max(), 1}},
                                        10)
              == model.aggregate_probability({{-2000, 1}, {2000, 1}}, 10),
            "level-10 raw out-of-range values affected bounded variance");

    require_throws<std::invalid_argument>(
      [&] { (void) model.aggregate_probability({{100, 1}}, -1); },
      "negative AggressiveLevel was accepted");
    require_throws<std::invalid_argument>(
      [&] { (void) model.aggregate_probability({{100, 1}}, 11); },
      "AggressiveLevel above 10 was accepted");

    require_throws<std::invalid_argument>(
      [&] { (void) model.aggregate_probability({{1, 0}, {2, 1}}, 3); },
      "zero probability count was accepted");
    require_throws<std::invalid_argument>(
      [&] {
          (void) model.aggregate_probability(
            {{1, ABJNNUE::Model::MaximumProbabilityCount + 1}}, 3);
      },
      "oversized probability count was accepted");
    require_throws<std::invalid_argument>(
      [&] {
          (void) model.aggregate_probability(
            {{1, ABJNNUE::Model::MaximumProbabilityCount}, {2, 1}}, 3);
      },
      "oversized probability total was accepted");
}

void test_visible_incremental(const ABJNNUE::Model& model) {
    std::deque<StateInfo> states(1);
    Position position;
    position.set(VisibleStartFEN, &states.back());
    auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
    compare_stack_with_full(model, position, *stack, "visible root");

    const Move move(SQ_A3, SQ_A4);
    states.emplace_back();
    const auto dirty = position.do_move(move, states.back(), position.gives_check(move), nullptr);
    stack->push(dirty, position);
    compare_stack_with_full(model, position, *stack, "visible move");
#if defined(ABJNNUE_RUNTIME_STATS)
    require(stack->stats().incrementalUpdates != 0,
            "visible move did not use incremental accumulation");
#endif

    position.undo_move(move);
    stack->pop();
    compare_stack_with_full(model, position, *stack, "visible undo");
}

void test_attack_and_midmirror_transitions(const ABJNNUE::Model& model) {
    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set(AttackBucketTransitionFEN, &states.back());
        require(ABJNNUE::FeatureEncoder::attack_bucket(position, WHITE) == 2,
                "attack-bucket fixture did not start with a visible rook");
        auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
        compare_stack_with_full(model, position, *stack, "attack-bucket root");

        const Move move(SQ_A9, SQ_A1);
        require(position.legal(move), "attack-bucket transition move is not legal");
        states.emplace_back();
        const auto dirty = position.do_move(move, states.back(), position.gives_check(move), nullptr);
        stack->push(dirty, position);
        require(ABJNNUE::FeatureEncoder::attack_bucket(position, WHITE) == 0,
                "attack-bucket transition did not remove White's rook bucket");
        compare_stack_with_full(model, position, *stack, "attack-bucket transition");

        position.undo_move(move);
        stack->pop();
        compare_stack_with_full(model, position, *stack, "attack-bucket undo");
    }

    {
        std::deque<StateInfo> states(1);
        Position position;
        position.set(MidMirrorTransitionFEN, &states.back());
        const bool before = ABJNNUE::FeatureEncoder::requires_mid_mirror(position, WHITE);
        auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
        compare_stack_with_full(model, position, *stack, "midmirror root");

        const Move move(SQ_A1, SQ_I1);
        require(position.legal(move), "MidMirror transition move is not legal");
        states.emplace_back();
        const auto dirty = position.do_move(move, states.back(), position.gives_check(move), nullptr);
        stack->push(dirty, position);
        const bool after = ABJNNUE::FeatureEncoder::requires_mid_mirror(position, WHITE);
        require(before != after, "MidMirror fixture did not cross its canonical symmetry boundary");
        compare_stack_with_full(model, position, *stack, "midmirror transition");

        position.undo_move(move);
        stack->pop();
        compare_stack_with_full(model, position, *stack, "midmirror undo");
    }
}

void test_hidden_capture_reveal(const ABJNNUE::Model& model) {
    std::deque<StateInfo> states(1);
    Position position;
    position.set(HiddenCaptureFEN, &states.back());
    auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
    compare_stack_with_full(model, position, *stack, "hidden root");

    const Move move(SQ_A3, SQ_A4);
    states.emplace_back();
    const auto dirty = position.do_move(move, states.back(), false, nullptr);
    stack->push(dirty, position);

    auto revealDirty = stack->latest_dirty_piece();
    const Piece hiddenPiece = position.do_flip(SQ_A4, W_PAWN, &revealDirty, nullptr);
    stack->push(revealDirty, position);
    compare_stack_with_full(model, position, *stack, "hidden capture reveal");
    require(position.rest_piece(W_PAWN) == 0, "revealed identity was not removed from the pool");

    position.undo_flip(SQ_A4, hiddenPiece);
    stack->pop();
    position.undo_move(move);
    stack->pop();
    compare_stack_with_full(model, position, *stack, "hidden capture undo");
    require(position.rest_piece(W_PAWN) == 1, "undo did not restore the identity pool");
}

void test_multi_hidden_capture_reveal(const ABJNNUE::Model& model) {
    std::deque<StateInfo> states(1);
    Position position;
    position.set(MultiHiddenCaptureFEN, &states.back());
    auto stack = std::make_unique<ABJNNUE::AccumulatorStack>();
    compare_stack_with_full(model, position, *stack, "multi-hidden root");

    const Move move(SQ_A3, SQ_A4);
    states.emplace_back();
    const auto dirty = position.do_move(move, states.back(), false, nullptr);
    stack->push(dirty, position);

    auto revealDirty = stack->latest_dirty_piece();
    stack->pop();
    const Piece hiddenPiece = position.do_flip(SQ_A4, W_PAWN, &revealDirty, nullptr);
    stack->push(revealDirty, position);
    compare_stack_with_full(model, position, *stack, "multi-hidden capture reveal");
#if defined(ABJNNUE_RUNTIME_STATS)
    require(stack->stats().incrementalUpdates != 0,
            "multi-hidden reveal did not use combined incremental accumulation");
#endif

    position.undo_flip(SQ_A4, hiddenPiece);
    position.undo_move(move);
    stack->pop();
    compare_stack_with_full(model, position, *stack, "multi-hidden capture undo");
}

void test_hidden_capture_chance_routing(const char* fen,
                                        Move        move,
                                        bool        expectMovingDark,
                                        Color       rootObserver,
                                        bool        expectChance,
                                        const char* label) {
    std::deque<StateInfo> states(1);
    Position position;
    position.set(fen, &states.back());

    require(position.move_dark(move) == expectMovingDark,
            std::string(label) + ": moving-piece visibility differs");
    const bool capturedDark = position.capture(move) && position.is_dark(move.to_sq());
    require(capturedDark, std::string(label) + ": fixture does not capture a dark piece");

    states.emplace_back();
    position.do_move(move, states.back(), position.gives_check(move), nullptr);

    Position::RestPieceList candidates{};
    const int candidateCount = Search::hidden_capture_candidates(
      position, rootObserver, capturedDark, candidates);
    require((candidateCount != 0) == expectChance,
            std::string(label) + ": hidden-capture chance routing differs");
    if (expectChance)
    {
        require(candidateCount == 2, std::string(label) + ": candidate type count differs");
        require(candidates[0] == std::pair<Piece, int>{B_ROOK, 2},
                std::string(label) + ": rook branch weight differs");
        require(candidates[1] == std::pair<Piece, int>{B_PAWN, 3},
                std::string(label) + ": pawn branch weight differs");
    }

    position.undo_move(move);
}

void test_hidden_capture_chance_routing() {
    test_hidden_capture_chance_routing(VisibleCapturesDarkFEN, Move(SQ_A5, SQ_A6), false,
                                       WHITE, true, "visible observer capture");
    test_hidden_capture_chance_routing(VisibleCapturesDarkFEN, Move(SQ_A5, SQ_A6), false,
                                       BLACK, false, "visible non-observer capture");
    test_hidden_capture_chance_routing(DarkCapturesDarkFEN, Move(SQ_A0, SQ_A6), true,
                                       WHITE, true, "dark observer capture");
    test_hidden_capture_chance_routing(DarkCapturesDarkFEN, Move(SQ_A0, SQ_A6), true,
                                       BLACK, false, "dark non-observer capture");
}

void test_hidden_capture_wrapper_repeatability(Engine&      engine,
                                               const char*  fen,
                                               const char*  searchMove,
                                               const char*  label) {
    engine.set_position(fen, {});
    const std::string rootFen = engine.fen();
    std::vector<std::string> bestMoves;
    engine.set_on_bestmove(
      [&bestMoves](std::string_view bestMove, std::string_view) {
          bestMoves.emplace_back(bestMove);
      });

    for (int search = 0; search < 2; ++search)
    {
        Search::LimitsType limits;
        limits.depth       = 2;
        limits.searchmoves = {searchMove};
        engine.go(limits);
        engine.wait_for_search_finished();

        require(engine.fen() == rootFen,
                std::string(label) + ": root board or hidden-piece pool changed after search");
        require(bestMoves.size() == std::size_t(search + 1),
                std::string(label) + ": search did not report exactly one best move");
        require(bestMoves.back().compare(0, 4, searchMove) == 0,
                std::string(label) + ": forced root move was not searched");
    }

    require(bestMoves[1] == bestMoves[0],
            std::string(label) + ": repeated search returned a different result");
    engine.set_on_bestmove([](std::string_view, std::string_view) {});
}

void test_hidden_capture_wrappers_are_repeatable(const std::filesystem::path& valid) {
    Engine engine(valid.string());
    engine.load_big_network(valid.string());
    engine.verify_networks();
    engine.set_on_update_no_moves([](const Engine::InfoShort&) {});
    engine.set_on_update_full([](const Engine::InfoFull&) {});
    engine.set_on_iter([](const Engine::InfoIter&) {});
    test_hidden_capture_wrapper_repeatability(engine, VisibleCapturesDarkFEN, "a5a6",
                                              "visible captures dark wrapper");
    test_hidden_capture_wrapper_repeatability(engine, DarkCapturesDarkFEN, "a0a6",
                                              "dark captures dark nested wrapper");
}

void test_observer_tt_isolation() {
    constexpr Key positionKey = 0x0123456789abcdefULL;
    const Key     whiteKey    = Search::observer_tt_key(positionKey, WHITE);
    const Key     blackKey    = Search::observer_tt_key(positionKey, BLACK);
    require(whiteKey != blackKey, "TT key does not isolate the root observer");
    require(Search::observer_tt_key(positionKey, WHITE) == whiteKey
              && Search::observer_tt_key(positionKey, BLACK) == blackKey,
            "observer TT key is not stable");
}

void test_failed_network_reload_is_recoverable(const std::filesystem::path& valid,
                                               const std::filesystem::path& invalid) {
    Engine engine(valid.string());
    engine.load_big_network(valid.string());
    engine.verify_networks();
    bool   rejected = false;
    engine.set_on_verify_networks([&rejected](std::string_view message) {
        rejected = message.find("ABJNNUE load failed:") != std::string_view::npos;
    });

    engine.load_big_network(std::filesystem::absolute(invalid).string());
    require(rejected, "legacy network reload did not report rejection");

    engine.load_big_network(valid.string());
    engine.verify_networks();

    // The public load API bypasses OptionsMap::operator=.  Loading the same
    // package through an absolute spelling must still update EvalFile, or
    // the next verify/search would see a filename/model mismatch.
    const auto absoluteValid = std::filesystem::absolute(valid).string();
    engine.load_big_network(absoluteValid);
    require(std::string(engine.get_options()["EvalFile"]) == absoluteValid,
            "direct network load did not synchronize EvalFile");
    engine.verify_networks();
}

void test_evalfile_option_defers_failed_reload_and_keeps_searchable_network(
  const std::filesystem::path& valid, const std::filesystem::path& invalid) {
    Engine engine(valid.string());
    engine.load_big_network(valid.string());
    engine.verify_networks();
    const std::string activeFile = engine.get_options()["EvalFile"];
    bool   rejected = false;
    engine.set_on_verify_networks([&rejected](std::string_view message) {
        rejected = message.find("ABJNNUE load failed:") != std::string_view::npos;
    });
    engine.set_on_update_no_moves([](const Engine::InfoShort&) {});
    engine.set_on_update_full([](const Engine::InfoFull&) {});
    engine.set_on_iter([](const Engine::InfoIter&) {});

    std::vector<std::string> bestMoves;
    engine.set_on_bestmove(
      [&bestMoves](std::string_view bestMove, std::string_view) {
          bestMoves.emplace_back(bestMove);
      });

    const auto rejectedFile = std::filesystem::absolute(invalid).string();
    std::istringstream command("name EvalFile value " + rejectedFile);
    engine.get_options().setoption(command);

    require(!rejected, "EvalFile option loaded a network before searching");
    require(std::string(engine.get_options()["EvalFile"]) == rejectedFile,
            "EvalFile option did not retain the pending selection");

    Search::LimitsType limits;
    limits.depth = 1;
    require_throws<std::runtime_error>([&] { engine.go(limits); },
                                       "search accepted a pending legacy EvalFile");
    require(std::string(engine.get_options()["EvalFile"]) == activeFile,
            "failed deferred EvalFile load replaced the active V11 network");
    require(bestMoves.empty(), "failed network load started a search");

    engine.go(limits);
    engine.wait_for_search_finished();
    require(bestMoves.size() == 1,
            "search did not complete after a rejected EvalFile change");
}

}  // namespace

int main(int argc, char** argv) {
    try
    {
        if (argc != 4)
            throw std::runtime_error(
              "usage: abjnnue_runtime_test abjchess-v11.nnue legacy-v6.nnue "
              "v11_feature_golden.json");

        Bitboards::init();
        Position::init();
        const auto model = ABJNNUE::Model::load(std::filesystem::path(argv[1]));
        require(model->inference_complete(), "runtime inference payload is incomplete");
        require(model->probability_complete(), "runtime probability payload is incomplete");
        const auto metadata = ABJNNUE::Package::load(std::filesystem::path(argv[1])).metadata_json();
        const bool smokeOnly = metadata.find("\"smoke_only\":true") != std::string::npos
                            || metadata.find("\"smoke_only\": true") != std::string::npos;

        test_heads(*model, !smokeOnly);
        test_head_backend_equivalence(*model);
        test_probability(*model, !smokeOnly);
        test_runtime_optimization_contract(*model);
        test_pairwise_transform_boundaries(*model);
        test_interpolation_oracle(*model);
        test_refresh_cache_hidden_and_fallback(*model);
        test_visible_incremental(*model);
        test_attack_and_midmirror_transitions(*model);
        test_hidden_capture_reveal(*model);
        test_multi_hidden_capture_reveal(*model);
        test_hidden_capture_chance_routing();
        test_v11_feature_golden(argv[3]);
        test_hidden_capture_wrappers_are_repeatable(argv[1]);
        test_observer_tt_isolation();
        test_failed_network_reload_is_recoverable(argv[1], argv[2]);
        test_evalfile_option_defers_failed_reload_and_keeps_searchable_network(argv[1], argv[2]);
        std::cout << "ABJNNUE runtime tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
