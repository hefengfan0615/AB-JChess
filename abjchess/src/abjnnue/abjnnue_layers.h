#ifndef ABJNNUE_LAYERS_H_INCLUDED
#define ABJNNUE_LAYERS_H_INCLUDED

#include <cstdint>

namespace ABJNNUE::Layers {

// Evaluates one V11 SFNN head: 2064->32, pair->64->32, concat->128->1 plus skip.
std::int32_t propagate(const std::uint8_t* head, const std::uint8_t* input);

// Bit-exact reference used to validate every architecture-specific path.
std::int32_t propagate_scalar(const std::uint8_t* head, const std::uint8_t* input);

const char* backend_name() noexcept;

}  // namespace ABJNNUE::Layers

#endif
