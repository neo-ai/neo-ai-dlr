#include "dlr.h"

#include "dlr_common.h"
#include "dlr_treelite.h"
#include "dlr_tvm.h"
#ifdef DLR_HEXAGON
#include "dlr_hexagon/dlr_hexagon.h"
#endif  // DLR_HEXAGON

#include <locale>

using namespace dlr;

/* DLR C API implementation */

extern "C" int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *num_inputs = model->GetNumInputs();
  API_END();
}

extern "C" int GetDLRNumWeights(DLRModelHandle* handle, int* num_weights) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *num_weights = model->GetNumWeights();
  API_END();
}

extern "C" int GetDLRInputName(DLRModelHandle* handle, int index,
                               const char** input_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *input_name = model->GetInputName(index);
  API_END();
}

extern "C" int GetDLRInputType(DLRModelHandle* handle, int index,
                               const char** input_type) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *input_type = model->GetInputType(index);
  API_END();
}

extern "C" int GetDLRWeightName(DLRModelHandle* handle, int index,
                                const char** weight_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *weight_name = model->GetWeightName(index);
  API_END();
}

extern "C" int SetDLRInput(DLRModelHandle* handle, const char* name,
                           const int64_t* shape, void* input, int dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetInput(name, shape, input, dim);
  API_END();
}

extern "C" int GetDLRInput(DLRModelHandle* handle, const char* name,
                           void* input) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetInput(name, input);
  API_END();
}

extern "C" int GetDLROutputShape(DLRModelHandle* handle, int index,
                                 int64_t* shape) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputShape(index, shape);
  API_END();
}

extern "C" int GetDLROutput(DLRModelHandle* handle, int index, void* out) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutput(index, out);
  API_END();
}

extern "C" int GetDLROutputSizeDim(DLRModelHandle* handle, int index,
                                   int64_t* size, int* dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputSizeDim(index, size, dim);
  API_END();
}

extern "C" int GetDLROutputType(DLRModelHandle* handle, int index,
                                const char** output_type) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *output_type = model->GetOutputType(index);
  API_END();
}

extern "C" int GetDLRNumOutputs(DLRModelHandle* handle, int* num_outputs) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *num_outputs = model->GetNumOutputs();
  API_END();
}

extern "C" int GetDLRHasMetadata(DLRModelHandle* handle, bool* has_metadata) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *has_metadata = model->HasMetadata();
  API_END();
}

extern "C" int GetDLROutputName(DLRModelHandle* handle, const int index, const char** name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *name = model->GetOutputName(index);
  try {
    *name = model->GetOutputName(index);
  } catch (dmlc::Error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  API_END();
}

extern "C" int GetDLROutputIndex(DLRModelHandle* handle, const char* name, int* index) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  try {
    *index = model->GetOutputIndex(name);
  } catch (dmlc::Error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  API_END();
}

extern "C" int GetDLROutputByName(DLRModelHandle* handle, const char* name, void* out) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputByName(name, out);
  API_END();
}

#ifdef DLR_HEXAGON
/*! \brief Translate c args from ctypes to std types for DLRModelFromHexagon
 * ctor.
 */
int CreateDLRModelFromHexagon(DLRModelHandle* handle, const char* model_path,
                              int debug_level) {
  API_BEGIN();
  const std::string model_path_string(model_path);
  // HexagonModel class does not use DLContext internally
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(1);  // 1 - kDLCPU
  ctx.device_id = 0;
  DLRModel* model = new HexagonModel(model_path_string, ctx, debug_level);
  *handle = model;
  API_END();
}
#endif  // DLR_HEXAGON

/*! \brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" int CreateDLRModel(DLRModelHandle* handle, const char* model_path,
                              int dev_type, int dev_id) {
  API_BEGIN();
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = dev_id;

  /* Logic to handle Windows drive letter */
  std::string model_path_string{model_path};
  std::string special_prefix{""};
  if (model_path_string.length() >= 2 && model_path_string[1] == ':' &&
      std::isalpha(model_path_string[0], std::locale("C"))) {
    // Handle drive letter
    special_prefix = model_path_string.substr(0, 2);
    model_path_string = model_path_string.substr(2);
  }

  std::vector<std::string> path_vec = dmlc::Split(model_path_string, ':');
  path_vec[0] = special_prefix + path_vec[0];

  DLRBackend backend = dlr::GetBackend(path_vec);
  DLRModel* model;
  if (backend == DLRBackend::kTVM) {
    model = new TVMModel(path_vec, ctx);
  } else if (backend == DLRBackend::kTREELITE) {
    model = new TreeliteModel(path_vec, ctx);
#ifdef DLR_HEXAGON
  } else if (backend == DLRBackend::kHEXAGON) {
    DLRModelHandle hexagon_handle;
    int errC = CreateDLRModelFromHexagon(&hexagon_handle, model_path,
                                         1 /*debug_level*/);
    if (errC != 0) return errC;
    model = static_cast<DLRModel*>(hexagon_handle);
#endif  // DLR_HEXAGON
  } else {
    LOG(FATAL) << "Unsupported backend!";
    return -1;  // unreachable
  }
  *handle = model;
  API_END();
}

extern "C" int DeleteDLRModel(DLRModelHandle* handle) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  delete model;
  *handle = NULL;
  API_END();
}

extern "C" int RunDLRModel(DLRModelHandle* handle) {
  API_BEGIN();
  static_cast<DLRModel*>(*handle)->Run();
  API_END();
}

extern "C" const char* DLRGetLastError() { return TVMGetLastError(); }

extern "C" int GetDLRBackend(DLRModelHandle* handle, const char** name) {
  API_BEGIN();
  *name = static_cast<DLRModel*>(*handle)->GetBackend();
  API_END();
}

extern "C" int GetDLRVersion(const char** out) {
  API_BEGIN();
  std::string version_str = std::to_string(DLR_MAJOR) + "." +
                            std::to_string(DLR_MINOR) + "." +
                            std::to_string(DLR_PATCH);
  *out = version_str.c_str();
  API_END();
}

extern "C" int SetDLRNumThreads(DLRModelHandle* handle, int threads) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetNumThreads(threads);
  API_END();
}

extern "C" int UseDLRCPUAffinity(DLRModelHandle* handle, int use) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->UseCPUAffinity(use);
  API_END();
}
