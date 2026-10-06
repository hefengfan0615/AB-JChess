#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <string>

#include "../src/abjnnue/abjnnue_package.h"

namespace {

void expect(std::string_view input, std::string_view expected) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(input.data());
    const auto actual = ABJNNUE::sha256_hex({bytes, input.size()});
    if (actual != expected)
        throw std::runtime_error("SHA-256 mismatch for test vector: " + std::string(input));
}

}  // namespace

int main() {
    try {
        expect("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        expect("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        expect("The quick brown fox jumps over the lazy dog",
               "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592");
        std::string millionA(1'000'000, 'a');
        expect(millionA,
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        std::cout << "ABJNNUE SHA-256 vectors passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
