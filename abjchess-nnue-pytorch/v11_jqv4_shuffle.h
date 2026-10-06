#pragma once

#include "jqv4/jqv4_rng.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace abjchess::v11_training {

class BoundedOrdinalShuffleV1
{
public:
    BoundedOrdinalShuffleV1(
        std::uint64_t count, std::uint32_t buffer_size, std::uint64_t seed)
        : m_rng(jieqi::v4::Xoshiro256StarStarV1::from_seed(seed)),
          m_next_ordinal(std::min<std::uint64_t>(count, buffer_size)),
          m_count(count)
    {
        if (buffer_size == 0U)
            throw std::invalid_argument(
                "bounded ordinal shuffle buffer must be positive");
        const auto initial_size = static_cast<std::size_t>(
            std::min<std::uint64_t>(count, buffer_size));
        m_window.resize(initial_size);
        std::iota(m_window.begin(), m_window.end(), UINT64_C(0));
    }

    std::optional<std::uint64_t> next()
    {
        if (m_window.empty())
            return std::nullopt;
        const auto selected = m_rng.bounded_u64(
            static_cast<std::uint64_t>(m_window.size()));
        if (!selected.ok())
            throw std::runtime_error(
                "bounded ordinal shuffle RNG failed: "
                + selected.status.message);
        const auto slot = static_cast<std::size_t>(*selected.value);
        const auto result = m_window[slot];
        if (m_next_ordinal < m_count) {
            m_window[slot] = m_next_ordinal++;
        } else {
            m_window[slot] = m_window.back();
            m_window.pop_back();
        }
        return result;
    }

private:
    jieqi::v4::Xoshiro256StarStarV1 m_rng;
    std::vector<std::uint64_t> m_window;
    std::uint64_t m_next_ordinal;
    std::uint64_t m_count;
};

inline std::vector<std::uint64_t>
permuted_ordinals_v1(std::size_t count, std::uint64_t seed)
{
    std::vector<std::uint64_t> result(count);
    std::iota(result.begin(), result.end(), UINT64_C(0));
    const auto shuffled = jieqi::v4::fisher_yates_v1(result, seed);
    if (!shuffled.ok())
        throw std::runtime_error(
            "ordinal Fisher-Yates shuffle failed: " + shuffled.message);
    return result;
}

} // namespace abjchess::v11_training
