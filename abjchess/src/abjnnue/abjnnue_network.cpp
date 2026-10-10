#include "abjnnue_network.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#endif

namespace Stockfish::Eval::NNUE {

namespace {

std::filesystem::path native_path(const std::string& value) {
    // MinGW's narrow argv uses the active Windows code page, while
    // libstdc++ filesystem expects UTF-8 for narrow paths.  Prefer UTF-8
    // (which is what UCI clients normally send), then fall back to decoding
    // the process/console spelling from the active code page.
    try
    {
        return std::filesystem::u8path(value);
    }
    catch (const std::filesystem::filesystem_error&)
    {
#ifdef _WIN32
        if (!value.empty())
        {
            // argv[0] and paths entered by a legacy GUI are often encoded in
            // the active code page (for example CP936), while libstdc++'s
            // narrow filesystem API requires UTF-8.  Decode ACP to wide,
            // then explicitly encode UTF-8 before constructing the path;
            // constructing path(std::wstring) directly invokes the same
            // locale conversion that raised the original exception.
            int length = MultiByteToWideChar(CP_ACP, MB_ERR_INVALID_CHARS,
                                             value.data(), int(value.size()), nullptr, 0);
            if (length <= 0)
                length = MultiByteToWideChar(CP_ACP, 0, value.data(), int(value.size()), nullptr, 0);
            if (length > 0)
            {
                std::wstring wide(std::size_t(length), L'\0');
                if (MultiByteToWideChar(CP_ACP, 0, value.data(), int(value.size()), wide.data(),
                                        length)
                    > 0)
                {
                    const int utf8Length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                                               wide.data(), int(wide.size()),
                                                               nullptr, 0, nullptr, nullptr);
                    if (utf8Length > 0)
                    {
                        std::string utf8(std::size_t(utf8Length), '\0');
                        if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(),
                                                int(wide.size()), utf8.data(), utf8Length,
                                                nullptr, nullptr)
                            > 0)
                            return std::filesystem::u8path(utf8);
                    }
                }
            }
        }
#endif
        throw;
    }
}

std::vector<std::filesystem::path> candidate_paths(const std::string& rootDirectory,
                                                   const std::string& evalFile) {
    const std::filesystem::path requested = native_path(evalFile);
    if (requested.is_absolute())
        return {requested};

    std::vector<std::filesystem::path> paths;
    if (!rootDirectory.empty())
        paths.emplace_back(native_path(rootDirectory) / requested);
    paths.emplace_back(requested);
    return paths;
}

}  // namespace

void NetworkBig::load(const std::string& rootDirectory, const std::string& evalFile) {
    if (evalFile.empty())
        throw std::invalid_argument("EvalFile must name a V11 .nnue package");

    std::ostringstream missing;
    bool first = true;
    for (const auto& candidate : candidate_paths(rootDirectory, evalFile))
    {
        if (!std::filesystem::exists(candidate))
        {
            if (!first)
                missing << ", ";
            try
            {
                missing << candidate.u8string();
            }
            catch (const std::exception&)
            {
                missing << "<network path>";
            }
            first = false;
            continue;
        }

        auto loaded = ::ABJNNUE::Model::load(candidate);
        model_ = std::move(loaded);
        currentFile_ = evalFile;
        return;
    }

    throw std::runtime_error("ABJCHESSV11 file not found; tried: " + missing.str());
}

void NetworkBig::verify(const std::string& requested,
                        const std::function<void(std::string_view)>& reporter) const {
    if (!model_ || currentFile_ != requested)
        throw std::runtime_error("ABJCHESSV11 package was not loaded for EvalFile=" + requested);
    if (reporter)
        reporter(model_->summary());
}

const ::ABJNNUE::Model& NetworkBig::model() const {
    if (!model_)
        throw std::logic_error("ABJNNUE model is unavailable");
    return *model_;
}

}  // namespace Stockfish::Eval::NNUE
