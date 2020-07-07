#ifndef DLR_COMMON_H_
#define DLR_COMMON_H_

#include <dmlc/common.h>
#include <dmlc/logging.h>
#include <runtime_base.h>
#include <sys/types.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#define LINE_SIZE 256

#ifndef _WIN32

/* For basename() on UNIX */
#include <libgen.h>

#endif

/* OS-specific library file extension and the name of the generated dynamic
 * library*/
#ifdef _WIN32
#define LIBEXT ".dll"
#define LIBDLR "dlr.dll"
#elif __APPLE__
#define LIBEXT ".dylib"
#define LIBDLR "libdlr.dylib"
#else
#define LIBEXT ".so"
#define LIBDLR "libdlr.so"
#endif

namespace dlr {

/* The following file names are reserved by SageMaker and should not be used
 * as model JSON */
constexpr const char *SAGEMAKER_AUXILIARY_JSON_FILES[] = {"model-shapes.json",
                                                          "hyperparams.json"};

struct ModelArtifact {
  std::string ver_json;
  std::string metadata;
};

bool IsFileEmpty(const std::string &filePath);

std::vector<std::string> ListFilesInDirectory(
    const std::string &directory_path);

std::vector<std::string> ListFilesInDirectories(
    const std::vector<std::string> &directory_paths);

std::string GetBasename(const std::string &path);

std::string GetParentFolder(const std::string &path);

void LoadJsonFromFile(const std::string &path, nlohmann::json &jsonObject);

inline bool StartsWith(const std::string &mainStr, const std::string &toMatch) {
  return mainStr.size() >= toMatch.size() &&
         mainStr.compare(0, toMatch.size(), toMatch) == 0;
}

inline bool EndsWith(const std::string &mainStr, const std::string &toMatch) {
  if (mainStr.size() >= toMatch.size() &&
      mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(),
                      toMatch) == 0)
    return true;
  else
    return false;
}

enum class DLRBackend { kTVM, kTREELITE, kTFLITE, kTENSORFLOW, kHEXAGON };

/*! \brief Get the backend based on the contents of the model folder.
 */
DLRBackend GetBackend(std::vector<std::string> dirname);

#define CHECK_SHAPE(msg, value, expected) \
  CHECK_EQ(value, expected)               \
      << (msg) << ". Value read: " << (value) << ", Expected: " << (expected);

}  // namespace dlr

#endif  // DLR_COMMON_H_
