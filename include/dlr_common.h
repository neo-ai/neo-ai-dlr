#ifndef DLR_COMMON_H_
#define DLR_COMMON_H_

#include <dmlc/common.h>
#include <dmlc/logging.h>
#include <runtime_base.h>
#include <sys/types.h>

#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

#include "dlr_allocator.h"

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

#ifndef DLR_MODEL_ELEM
#define DLR_MODEL_ELEM
enum DLRModelElemType {
  HEXAGON_LIB,
  NEO_METADATA,
  TVM_GRAPH,
  TVM_LIB,
  TVM_PARAMS,
  RELAY_EXEC,
  TF2_SAVED_MODEL
};
typedef struct ModelElem {
  const DLRModelElemType type;
  const char* path;
  const void* data;
  const size_t data_size;
} DLRModelElem;
#endif

namespace dlr {

/* The following file names are reserved by SageMaker and should not be used
 * as model JSON */
constexpr const char* SAGEMAKER_AUXILIARY_JSON_FILES[] = {"model-shapes.json", "hyperparams.json"};

typedef struct {
  std::string model_lib;
  std::string params;
  std::string model_json;
  std::string metadata;
  std::string relay_executable;
} ModelPath;

void ListDir(const std::string& path, std::vector<std::string>& paths);

DLR_DLL std::vector<std::string> FindFiles(const std::vector<std::string>& paths);

/* Logic to handle Windows drive letter */
std::string FixWindowsDriveLetter(const std::string& path);

std::string GetBasename(const std::string& path);

bool IsFileEmpty(const std::string& filePath);

std::string GetParentFolder(const std::string& path);

DLR_DLL void LoadJsonFromString(const std::string& jsonData, nlohmann::json& jsonObject);
DLR_DLL void LoadJsonFromFile(const std::string& path, nlohmann::json& jsonObject);

DLR_DLL std::string LoadFileToString(const std::string& path,
                                     std::ios_base::openmode mode = std::ios_base::in);

inline bool StartsWith(const std::string& mainStr, const std::string& toMatch) {
  return mainStr.size() >= toMatch.size() && mainStr.compare(0, toMatch.size(), toMatch) == 0;
}

inline bool EndsWith(const std::string& mainStr, const std::string& toMatch) {
  if (mainStr.size() >= toMatch.size() &&
      mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
    return true;
  else
    return false;
}

enum class DLRBackend { kTVM, kTREELITE, kHEXAGON, kRELAYVM, kPIPELINE, kUNKNOWN, kTENSORFLOW2 };
extern const char* kBackendToStr[7];

/*! \brief Get the backend based on the contents of the model folder.
 */
DLRBackend GetBackend(const std::vector<std::string>& files);
DLRBackend GetBackend(const std::vector<DLRModelElem>& model_elems);

void InitModelPath(const std::vector<std::string>& files, ModelPath* paths);

std::string GetMetadataFile(const std::string& dirname);

DLDeviceType GetDeviceTypeFromString(const std::string& target_backend);

std::string GetStringFromDeviceType(DLDeviceType device_type);

DLDeviceType GetDeviceTypeFromMetadata(const std::vector<std::string>& model_paths);

DLR_DLL std::vector<std::string> MakePathVec(std::string model_path);

bool HasNegative(const int64_t* arr, const size_t size);

#define CHECK_SHAPE(msg, value, expected) \
  CHECK_EQ(value, expected) << (msg) << ". Value read: " << (value) << ", Expected: " << (expected);

// Abstract class
class DLR_DLL DLRModel {
 protected:
  std::string version_;
  DLRBackend backend_;
  size_t num_inputs_ = 1;
  size_t num_weights_ = 0;
  size_t num_outputs_ = 1;
  DLDevice dev_;
  std::vector<std::string> input_names_;
  std::vector<std::string> input_types_;
  std::vector<std::vector<int64_t>> input_shapes_;
  virtual void ValidateDeviceTypeIfExists();

 public:
  nlohmann::json metadata_ = nullptr;
  DLRModel(const DLDevice& dev, const DLRBackend& backend) : dev_(dev), backend_(backend) {}
  virtual ~DLRModel() {}

  /* Input related functions */
  virtual int GetNumInputs() const { return num_inputs_; }
  virtual const char* GetInputName(int index) const = 0;
  virtual const char* GetInputType(int index) const = 0;
  virtual const int GetInputDim(int index) const = 0;
  virtual const int64_t GetInputSize(int index) const = 0;
  virtual const std::vector<int64_t>& GetInputShape(int index) const;
  virtual void GetInput(const char* name, void* input) = 0;
  virtual void SetInput(const char* name, const int64_t* shape, const void* input, int dim) = 0;

  /* Output related functions */
  virtual int GetNumOutputs() { return num_outputs_; }
  virtual const char* GetOutputName(const int index) const {
    throw dmlc::Error("GetOutputName is not supported for this model.");
    return nullptr;
  }
  virtual int GetOutputIndex(const char* name) const {
    throw dmlc::Error("GetOutputIndex is not supported for this model.");
    return -1;
  }
  virtual const char* GetOutputType(int index) const = 0;
  virtual void GetOutputShape(int index, int64_t* shape) const = 0;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) = 0;
  virtual void GetOutput(int index, void* out) = 0;
  virtual const void* GetOutputPtr(int index) const = 0;
  virtual void GetOutputByName(const char* name, void* out) {
    throw dmlc::Error("GetOutputByName is not supported yet!");
  }

  /* Weights related functions */
  virtual int GetNumWeights() const { return num_weights_; }
  virtual const char* GetWeightName(int index) const = 0;
  virtual std::vector<std::string> GetWeightNames() const = 0;

  virtual DLDeviceType GetDeviceTypeFromMetadata() const;
  virtual DLRBackend GetBackend() { return backend_; }
  virtual void SetNumThreads(int threads) = 0;
  virtual bool HasMetadata() const;
  virtual void UseCPUAffinity(bool use) = 0;
  virtual void Run() = 0;
};

typedef std::shared_ptr<DLRModel> DLRModelPtr;

}  // namespace dlr

#endif  // DLR_COMMON_H_
