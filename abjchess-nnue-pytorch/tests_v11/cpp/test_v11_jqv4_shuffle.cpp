#include "v11_jqv4_shuffle.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<std::uint64_t> collect(
    std::uint64_t count, std::uint32_t window, std::uint64_t seed)
{
    abjchess::v11_training::BoundedOrdinalShuffleV1 shuffle(
        count, window, seed);
    std::vector<std::uint64_t> result;
    while (const auto value = shuffle.next())
        result.push_back(*value);
    return result;
}

void test_bounded_order_is_deterministic_and_complete()
{
    const auto first = collect(1000U, 37U, 42U);
    const auto second = collect(1000U, 37U, 42U);
    const auto different = collect(1000U, 37U, 43U);
    require(first == second, "equal bounded-shuffle seeds differ");
    require(first != different, "different bounded-shuffle seeds match");
    require(first.size() == 1000U, "bounded shuffle lost ordinals");

    auto sorted = first;
    std::sort(sorted.begin(), sorted.end());
    for (std::uint64_t ordinal = 0U; ordinal < sorted.size(); ++ordinal)
        require(sorted[ordinal] == ordinal,
                "bounded shuffle duplicated or skipped an ordinal");

    for (std::uint64_t emitted = 0U; emitted < first.size(); ++emitted)
        require(first[emitted] <= emitted + 36U,
                "bounded shuffle selected beyond its lookahead window");
}

void test_bounded_order_handles_edges()
{
    require(collect(0U, 1U, 42U).empty(), "empty source emitted data");
    require(collect(3U, 4096U, 42U).size() == 3U,
            "oversized window lost data");
    bool rejected = false;
    try {
        (void)collect(1U, 0U, 42U);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "zero bounded-shuffle window was accepted");
}

void test_fisher_yates_helper_is_deterministic_and_complete()
{
    const auto first = abjchess::v11_training::permuted_ordinals_v1(256U, 99U);
    const auto second = abjchess::v11_training::permuted_ordinals_v1(256U, 99U);
    const auto different = abjchess::v11_training::permuted_ordinals_v1(256U, 100U);
    require(first == second, "equal Fisher-Yates seeds differ");
    require(first != different, "different Fisher-Yates seeds match");
    auto sorted = first;
    std::sort(sorted.begin(), sorted.end());
    for (std::uint64_t ordinal = 0U; ordinal < sorted.size(); ++ordinal)
        require(sorted[ordinal] == ordinal,
                "Fisher-Yates permutation is not complete");
}

} // namespace

int main()
{
    try {
        test_bounded_order_is_deterministic_and_complete();
        test_bounded_order_handles_edges();
        test_fisher_yates_helper_is_deterministic_and_complete();
        std::cout << "v11 jqv4 bounded shuffle tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "v11 jqv4 bounded shuffle test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
