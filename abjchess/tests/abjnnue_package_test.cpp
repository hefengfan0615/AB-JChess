#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "../src/abjnnue/abjnnue_package.h"
#include "../src/abjnnue/abjnnue_runtime_layout.h"

namespace {

template<typename PackageT, typename = void>
struct accepts_hash_bypass : std::false_type {};

template<typename PackageT>
struct accepts_hash_bypass<
  PackageT,
  std::void_t<decltype(PackageT::load(std::declval<const std::filesystem::path&>(), false))>>
    : std::true_type {};

static_assert(!accepts_hash_bypass<ABJNNUE::Package>::value,
              "Package::load must not expose a hash-verification bypass");

template<typename LayoutT, typename = void>
struct binds_temporary_package : std::false_type {};

template<typename LayoutT>
struct binds_temporary_package<
  LayoutT,
  std::void_t<decltype(LayoutT::bind(std::declval<ABJNNUE::Package&&>()))>>
    : std::true_type {};

static_assert(!binds_temporary_package<ABJNNUE::RuntimeLayout>::value,
              "RuntimeLayout::bind must reject temporary packages");

template<typename LayoutT, typename = void>
struct binds_const_temporary_package : std::false_type {};

template<typename LayoutT>
struct binds_const_temporary_package<
  LayoutT,
  std::void_t<decltype(LayoutT::bind(std::declval<const ABJNNUE::Package&&>()))>>
    : std::true_type {};

static_assert(!binds_const_temporary_package<ABJNNUE::RuntimeLayout>::value,
              "RuntimeLayout::bind must reject const temporary packages");

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open V11 package: " + path.string());
    const auto end = in.tellg();
    if (end == std::ifstream::pos_type(-1)) throw std::runtime_error("cannot stat V11 package");
    const auto size = static_cast<std::size_t>(end);
    std::vector<std::uint8_t> bytes(size);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) throw std::runtime_error("cannot read V11 package");
    return bytes;
}

void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create V11 package fixture");
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write V11 package fixture");
}

void expect_rejected(const std::filesystem::path& path, const char* name) {
    try
    {
        (void) ABJNNUE::Package::load(path);
    }
    catch (const std::exception&)
    {
        return;
    }
    throw std::runtime_error(std::string("accepted invalid V11 package: ") + name);
}

void expect_rejected_with_message(const std::filesystem::path& path,
                                  const char* name,
                                  std::string_view message) {
    try
    {
        (void) ABJNNUE::Package::load(path);
    }
    catch (const std::exception& error)
    {
        if (std::string_view(error.what()).find(message) != std::string_view::npos)
            return;
        throw std::runtime_error(std::string("invalid V11 package failed for the wrong reason: ")
                                 + name + ": " + error.what());
    }
    throw std::runtime_error(std::string("accepted invalid V11 package: ") + name);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint32_t read_u32_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4)
        throw std::runtime_error("fixture header is truncated");
    return std::uint32_t(bytes[offset]) | (std::uint32_t(bytes[offset + 1]) << 8)
         | (std::uint32_t(bytes[offset + 2]) << 16) | (std::uint32_t(bytes[offset + 3]) << 24);
}

std::size_t payload_offset(const std::vector<std::uint8_t>& bytes) {
    return ABJNNUE::Package::HeaderSize + read_u32_le(bytes, 20);
}

std::string metadata_json(const std::vector<std::uint8_t>& bytes) {
    const auto metadataLength = read_u32_le(bytes, 20);
    if (bytes.size() < ABJNNUE::Package::HeaderSize + metadataLength)
        throw std::runtime_error("fixture metadata is truncated");
    return std::string(reinterpret_cast<const char*>(bytes.data() + ABJNNUE::Package::HeaderSize),
                       metadataLength);
}

void replace_once(std::string& text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    if (pos == std::string::npos)
        throw std::runtime_error("fixture metadata does not contain: " + from);
    if (text.find(from, pos + from.size()) != std::string::npos)
        throw std::runtime_error("fixture metadata replacement is ambiguous: " + from);
    text.replace(pos, from.size(), to);
}

void write_metadata(std::vector<std::uint8_t>& bytes, const std::string& metadata) {
    const auto metadataLength = read_u32_le(bytes, 20);
    if (metadata.size() != metadataLength)
        throw std::runtime_error("fixture metadata mutation changed its length");
    std::copy(metadata.begin(), metadata.end(),
              bytes.begin() + static_cast<std::ptrdiff_t>(ABJNNUE::Package::HeaderSize));
}

void replace_chunk_size(std::string& metadata, const char* chunkName,
                        std::uint64_t oldSize, std::uint64_t newSize) {
    const auto namePos = metadata.find(std::string("\"name\":\"") + chunkName + "\"");
    if (namePos == std::string::npos)
        throw std::runtime_error("fixture metadata is missing chunk: " + std::string(chunkName));
    const std::string oldField = "\"size\":" + std::to_string(oldSize);
    const auto sizePos = metadata.find(oldField, namePos);
    if (sizePos == std::string::npos)
        throw std::runtime_error("fixture metadata is missing chunk size for: " + std::string(chunkName));
    metadata.replace(sizePos, oldField.size(), "\"size\":" + std::to_string(newSize));
}

void replace_chunk_hash(std::string& metadata, const char* chunkName,
                        const std::string& newHash) {
    const auto namePos = metadata.find(std::string("\"name\":\"") + chunkName + "\"");
    if (namePos == std::string::npos)
        throw std::runtime_error("fixture metadata is missing chunk: " + std::string(chunkName));
    const std::string marker = "\"sha256\":\"";
    const auto hashStart = metadata.find(marker, namePos);
    if (hashStart == std::string::npos)
        throw std::runtime_error("fixture metadata is missing chunk hash for: " + std::string(chunkName));
    const auto valueStart = hashStart + marker.size();
    const auto valueEnd = metadata.find('"', valueStart);
    if (valueEnd == std::string::npos)
        throw std::runtime_error("fixture metadata has an unterminated chunk hash");
    metadata.replace(valueStart, valueEnd - valueStart, newHash);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::cerr << "usage: abjnnue_package_test v11.nnue [legacy-v6.nnue] [legacy-v9.nnue]\n";
        return 2;
    }

    const std::filesystem::path validPath = argv[1];
    const auto fixture = std::filesystem::temp_directory_path() / "abjnnue_v11_package_test.nnue";
    try
    {
        const auto package = ABJNNUE::Package::load(validPath);
        const auto layout = ABJNNUE::RuntimeLayout::bind(package);
        require(package.version() == 110, "valid V11 package version is not 110");
        require(package.version() == ABJNNUE::Package::Version,
                "valid V11 package version differs from native ABI");
        require(package.chunks().size() == 4,
                "valid V11 package did not expose exactly four chunks");
        require(package.payload_size() == 143665528,
                "valid V11 package payload size is incorrect");
        require(layout.inference_complete(), "valid V11 inference payload is incomplete");
        require(layout.probability_complete(), "valid V11 probability payload is incomplete");
        require(package.metadata_json().find(ABJNNUE::Package::Schema) != std::string::npos
                  && package.metadata_json().find(ABJNNUE::Package::FeatureIdentity) != std::string::npos,
                "valid V11 package metadata is missing its schema identity");
        require(!package.has_chunk("aux_runtime_weight_payload.bin"),
                "valid V11 package exposes removed auxiliary chunk");

        const struct {
            const char*   name;
            std::uint64_t offset;
            std::uint64_t size;
        } expectedChunks[] = {
          {"primary_runtime_nnue_container.bin", 0, ABJNNUE::RuntimeLayout::PrimarySize},
          {"eval_heads_runtime.bin", 142545216, ABJNNUE::RuntimeLayout::EvalHeadsSize},
          {"probability_score_to_mass.i32le",
           143641920,
           ABJNNUE::RuntimeLayout::ProbabilityScoreToMassSize * sizeof(std::int32_t)},
          {"probability_mass_to_score.i32le",
           143657924,
           ABJNNUE::RuntimeLayout::ProbabilityMassToScoreSize * sizeof(std::int32_t)},
        };
        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto& chunk = package.chunks()[i];
            require(chunk.name == expectedChunks[i].name, "valid V11 chunk order mismatch");
            require(chunk.dataOffset == expectedChunks[i].offset, "valid V11 chunk offset mismatch");
            require(chunk.size == expectedChunks[i].size, "valid V11 chunk size mismatch");
            require(package.view(chunk).size == expectedChunks[i].size,
                    "valid V11 chunk view size mismatch");
        }

        auto bytes = read_file(validPath);
        if (bytes.size() < ABJNNUE::Package::HeaderSize)
            throw std::runtime_error("valid V11 package is shorter than its header");
        const std::uint8_t expectedMagic[16] = {
          'A', 'B', 'J', 'C', 'H', 'E', 'S', 'S', 'V', '1', '1', 0, 0, 0, 0, 0};
        require(std::equal(expectedMagic, expectedMagic + 16, bytes.begin()),
                "valid V11 package magic differs from ABJCHESSV11");

        // Every mutation below is written to a temporary fixture, so the supplied
        // production/smoke package remains byte-for-byte unchanged.
        auto mutated = bytes;
        mutated[0] ^= 1;
        write_file(fixture, mutated);
        expect_rejected(fixture, "bad magic");

        mutated = bytes;
        mutated[16] = 7;  // little-endian package version
        write_file(fixture, mutated);
        expect_rejected(fixture, "wrong package version");

        mutated = bytes;
        mutated[10] = '0';
        mutated[16] = 100;
        mutated[17] = 0;
        mutated[18] = 0;
        mutated[19] = 0;
        write_file(fixture, mutated);
        expect_rejected(fixture, "V10 package identity");

        mutated = bytes;
        const auto metadataLength = read_u32_le(mutated, 20);
        const auto payloadOffset = ABJNNUE::Package::HeaderSize + metadataLength;
        if (payloadOffset >= mutated.size()) throw std::runtime_error("V11 fixture has no payload");
        mutated[payloadOffset] ^= 1;
        write_file(fixture, mutated);
        expect_rejected(fixture, "chunk SHA-256 corruption");

        mutated = bytes;
        mutated.resize(mutated.size() - 1);
        write_file(fixture, mutated);
        expect_rejected(fixture, "truncated package");

        mutated = bytes;
        mutated.push_back(0);
        write_file(fixture, mutated);
        expect_rejected_with_message(fixture, "trailing payload byte",
                                     "ABJCHESSV11 payload size mismatch");

        mutated = bytes;
        auto metadata = metadata_json(mutated);
        const auto v81PayloadOffset = payload_offset(mutated);
        const auto primaryEnd = v81PayloadOffset + ABJNNUE::RuntimeLayout::PrimarySize;
        if (primaryEnd > mutated.size()) throw std::runtime_error("V11 fixture has no primary payload");
        mutated.erase(mutated.begin() + static_cast<std::ptrdiff_t>(primaryEnd - 1));
        replace_chunk_size(metadata, "primary_runtime_nnue_container.bin",
                           ABJNNUE::RuntimeLayout::PrimarySize,
                           ABJNNUE::RuntimeLayout::PrimarySize - 1);
        replace_once(metadata, "\"data_offset\":142545216", "\"data_offset\":142545215");
        replace_once(metadata, "\"data_offset\":143641920", "\"data_offset\":143641919");
        replace_once(metadata, "\"data_offset\":143657924", "\"data_offset\":143657923");
        replace_chunk_hash(metadata, "primary_runtime_nnue_container.bin",
                           ABJNNUE::sha256_hex({mutated.data() + v81PayloadOffset,
                                                ABJNNUE::RuntimeLayout::PrimarySize - 1}));
        write_metadata(mutated, metadata);
        write_file(fixture, mutated);
        expect_rejected_with_message(fixture, "wrong primary chunk size",
                                     "ABJCHESSV11 chunk size mismatch");

        mutated = bytes;
        metadata = metadata_json(mutated);
        const auto massToScoreOffset = v81PayloadOffset + 143657924;
        if (massToScoreOffset > mutated.size())
            throw std::runtime_error("V11 fixture has no probability payload");
        mutated.insert(mutated.begin() + static_cast<std::ptrdiff_t>(massToScoreOffset), 0);
        replace_once(metadata, "\"data_offset\":143657924", "\"data_offset\":143657925");
        write_metadata(mutated, metadata);
        write_file(fixture, mutated);
        expect_rejected_with_message(fixture, "payload gap",
                                     "ABJCHESSV11 chunk layout mismatch");

        for (const auto& change : std::vector<std::pair<std::string, std::string>>{
              {ABJNNUE::Package::Schema, "abjchess-v00-sfnn-inventory-context-v1"},
              {ABJNNUE::Package::ArchitectureHash, std::string(64, '0')},
              {"\"psqt_buckets\":0", "\"psqt_buckets\":8"},
              {"eval_heads_runtime.bin", "eval_heads_missing.bin"}})
        {
            mutated = bytes;
            metadata = metadata_json(mutated);
            auto at = metadata.find(change.first);
            require(at != std::string::npos, "mutation field absent");
            metadata.replace(at, change.first.size(), change.second);
            write_metadata(mutated, metadata);
            write_file(fixture, mutated);
            expect_rejected(fixture, "contract mutation");
        }

        // A V6 file may be present beside the V11 package, but it must never be
        // accepted by the V11 parser or network loader.
        if (argc >= 3)
        {
            require(std::filesystem::is_regular_file(argv[2]),
                    "legacy V6 fixture is missing");
            expect_rejected(argv[2], "legacy V6 package");
        }
        if (argc >= 4)
        {
            require(std::filesystem::is_regular_file(argv[3]),
                    "legacy V9 fixture is missing");
            expect_rejected(argv[3], "legacy V9 package");
        }

        std::filesystem::remove(fixture);
        std::cout << "ABJNNUE V11 package tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::filesystem::remove(fixture);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
