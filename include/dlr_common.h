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

/* OS-specific library file extension and the name of the generated dynamic library*/
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

#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL __declspec(dllexport)
#else
#define DLR_DLL
#endif  // defined(_MSC_VER) || defined(_WIN32)

namespace dlr {

/* The following file names are reserved by SageMaker and should not be used
 * as model JSON */
constexpr const char* SAGEMAKER_AUXILIARY_JSON_FILES[] = {"model-shapes.json",
                                                          "hyperparams.json"};

typedef struct {
  std::string model_lib;
  std::string params;
  std::string model_json;
  std::string ver_json;
  std::string metadata;
} ModelPath;

void ListDir(const std::string& dirname, std::vector<std::string>& paths);

std::string GetBasename(const std::string& path);

std::string GetParentFolder(const std::string& path);

void LoadJsonFromFile(const std::string& path, nlohmann::json& jsonObject);

inline bool StartsWith(const std::string& mainStr, const std::string& toMatch) {
  return mainStr.size() >= toMatch.size() &&
  mainStr.compare(0, toMatch.size(), toMatch) == 0;
}

inline bool EndsWith(const std::string& mainStr, const std::string& toMatch) {
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

// Abstract class
class DLR_DLL DLRModel {
 protected:
  std::string version_;
  DLRBackend backend_;
  size_t num_inputs_ = 1;
  size_t num_weights_ = 0;
  size_t num_outputs_ = 1;
  DLContext ctx_;
  std::vector<std::string> input_names_;
  std::vector<std::string> input_types_;

 public:
  DLRModel(const DLContext& ctx, const DLRBackend& backend)
      : ctx_(ctx), backend_(backend) {}
  virtual ~DLRModel() {}

  /* Input related functions */
  virtual int GetNumInputs() const { return num_inputs_; }
  virtual const char* GetInputName(int index) const = 0;
  virtual const char* GetInputType(int index) const = 0;
  virtual void GetInput(const char* name, void* input) = 0;
  virtual void SetInput(const char* name, const int64_t* shape, void* input, int dim) = 0;

  /* Ouput related functions */
  virtual int GetNumOutputs() { return num_outputs_; }
  virtual const char* GetOutputName(const int index) const { 
    LOG(ERROR) << "GetOutputName is not supported yet!";
    return NULL;
  }
  virtual int GetOutputIndex(const char* name) const { 
    LOG(ERROR) << "GetOutputName is not supported yet!";
    return -1;
  }
  virtual const char* GetOutputType(int index) const = 0;
  virtual void GetOutputShape(int index, int64_t* shape) const = 0;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) = 0;
  virtual void GetOutput(int index, void* out) = 0;
  virtual void GetOutputByName(const char* name, void* out) {
    LOG(ERROR) << "GetOutputByName is not supported yet!";
  }
  
  /* Weights releated functions */
  virtual int GetNumWeights() const { return num_weights_; }
  virtual const char* GetWeightName(int index) const = 0;
  virtual std::vector<std::string> GetWeightNames() const = 0;
  
  virtual const char* GetBackend() const = 0;
  virtual void SetNumThreads(int threads) = 0;
  virtual bool HasMetadata() const {return false; }
  virtual void UseCPUAffinity(bool use) = 0;
  virtual void Run() = 0;
  
};

}  // namespace dlr

#endif  // DLR_COMMON_H_
