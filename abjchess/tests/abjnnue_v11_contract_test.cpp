#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "../src/abjnnue/abjnnue_package.h"
#include "../src/abjnnue/abjnnue_runtime_layout.h"

static_assert(ABJNNUE::Package::Version == 110,
              "the V11 contract test must compile against the V11 package ABI");
static_assert(ABJNNUE::RuntimeLayout::FeatureDimensions == 34800,
              "the V11 contract test must compile against the 24-bucket feature ABI");

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("usage: abjnnue_v11_contract_test v11.nnue");

        if (ABJNNUE::Package::Version != 110)
            throw std::runtime_error("V11 package version is not 110");
        if (ABJNNUE::RuntimeLayout::FeatureDimensions != 34800)
            throw std::runtime_error("V11 feature dimensions are not 34800");

        const auto package = ABJNNUE::Package::load(std::filesystem::path(argv[1]));
        const auto layout = ABJNNUE::RuntimeLayout::bind(package);
        if (!layout.inference_complete())
            throw std::runtime_error("V11 inference payload is incomplete");
        if (!layout.probability_complete())
            throw std::runtime_error("V11 probability payload is incomplete");

        std::cout << "ABJNNUE V11 contract passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
