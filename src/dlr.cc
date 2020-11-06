#include "dlr.h"

#include "dlr_common.h"
#include "dlr_pipeline.h"
#include "dlr_relayvm.h"
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

extern "C" int GetDLRInputShape(DLRModelHandle* handle, int index,
                                 int64_t* shape) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  const std::vector<int64_t>& input_shape = model->GetInputShape(index);
  std::copy(input_shape.begin(), input_shape.end(), shape);
  API_END();
}

extern "C" int GetDLRInputSizeDim(DLRModelHandle* handle, int index,
                                   int64_t* size, int* dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *size = model->GetInputSize(index);
  *dim = model->GetInputDim(index);
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
                           const int64_t* shape, const void* input, int dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetInput(name, shape, input, dim);
  API_END();
}

extern "C" int SetTVMInputTensor(DLRModelHandle* handle, const char* name,
                                 void* dltensor) {
  API_BEGIN();
  DLTensor* tensor = static_cast<DLTensor*>(dltensor);
  TVMModel* model = static_cast<TVMModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetInput(name, tensor);
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

extern "C" int GetDLROutputPtr(DLRModelHandle* handle, int index, const void** out) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *out = model->GetOutputPtr(index);
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

extern "C" int GetTVMOutputTensor(DLRModelHandle* handle, int index, void* dltensor) {
  API_BEGIN();
  DLTensor* tensor = static_cast<DLTensor*>(dltensor);
  TVMModel* model = static_cast<TVMModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->CopyOutputTensor(index, tensor);
  API_END();
}

std::vector<std::string> MakePathVec(const char* model_path) {
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
  return path_vec;
}

DLRModelPtr CreateDLRModelPtr(const char* model_path, DLContext& ctx) {
  std::vector<std::string> path_vec = MakePathVec(model_path);
  DLRBackend backend = dlr::GetBackend(path_vec);
  if (backend == DLRBackend::kTVM) {
    return std::make_shared<TVMModel>(path_vec, ctx);
  } else if (backend == DLRBackend::kRELAYVM) {
    return std::make_shared<RelayVMModel>(path_vec, ctx);
  } else if (backend == DLRBackend::kTREELITE) {
    return std::make_shared<TreeliteModel>(path_vec, ctx);
  #ifdef DLR_HEXAGON
    } else if (backend == DLRBackend::kHEXAGON) {
      const std::string model_path_string(model_path);
      return std::make_shared<HexagonModel>(model_path_string, ctx, 1 /*debug_level*/);
  #endif  // DLR_HEXAGON
  } else {
    throw dmlc::Error("Unsupported backend!");
  }
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

  std::vector<std::string> path_vec = MakePathVec(model_path);

  DLRBackend backend = dlr::GetBackend(path_vec);
  DLRModel* model;
  try {
    if (backend == DLRBackend::kTVM) {
      model = new TVMModel(path_vec, ctx);
    } else if (backend == DLRBackend::kRELAYVM) {
      model = new RelayVMModel(path_vec, ctx);
    } else if (backend == DLRBackend::kTREELITE) {
      model = new TreeliteModel(path_vec, ctx);
  #ifdef DLR_HEXAGON
    } else if (backend == DLRBackend::kHEXAGON) {
      const std::string model_path_string(model_path);
      model = new HexagonModel(model_path_string, ctx, 1 /*debug_level*/);
  #endif  // DLR_HEXAGON
    } else {
      LOG(FATAL) << "Unsupported backend!";
      return -1;  // unreachable
    }
  } catch (dmlc::Error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  
  *handle = model;
  API_END();
}

/*! \brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" int CreateDLRModelFromPaths(DLRModelHandle* handle, const DLRPaths* paths,
                                       int dev_type, int dev_id) {
  API_BEGIN();
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = dev_id;

  ModelPath path;
  path.model_lib = paths->model_lib;
  path.params = paths->params;
  path.model_json = paths->model_json;
  path.ver_json = paths->ver_json;
  path.metadata = paths->metadata;
  path.relay_executable = paths->relay_executable;
  
  DLRModel* model;
  try {
    if (paths->model_lib != NULL and paths->params != NULL and paths->model_json != NULL) {
      model = new TVMModel(path, ctx);
    } else if (paths->model_lib != NULL and paths->relay_executable != NULL) {
      model = new RelayVMModel(path, ctx);
    } else {
      LOG(FATAL) << "Unsupported backend!";
      return -1;  // unreachable
    }
  } catch (dmlc::Error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  
  *handle = model;
  API_END();
}

extern "C" int CreateTVMModel(DLRModelHandle* handle,
                              const char* graph,
                              const char* lib_path,
                              const char* params,
                              unsigned params_len,
                              int dev_type, int dev_id) {
  API_BEGIN();
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = dev_id;
  /* Logic to handle Windows drive letter */
  auto path_fix = [=](auto path) {
    std::string path_string{path};
    std::string special_prefix{""};
    if (path_string.length() >= 2 && path_string[1] == ':' &&
        std::isalpha(path_string[0], std::locale("C"))) {
      // Handle drive letter
      special_prefix = path_string.substr(0, 2);
      path_string = path_string.substr(2);
    }
    return special_prefix + path_string;
  };
  ModelPath paths;
  paths.model_lib = path_fix(lib_path);
  std::string param_str(params,params_len);
  *handle = new TVMModel(graph, param_str, paths, ctx);
  API_END();
}

/*! \brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" int CreateDLRPipeline(DLRModelHandle* handle,
                                 int num_models, const char** model_paths,
                                 int dev_type, int dev_id) {
  API_BEGIN();
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = dev_id;
  std::vector<DLRModelPtr> dlr_models;
  for (int i = 0; i < num_models; i++) {
    try {
      DLRModelPtr model_ptr = CreateDLRModelPtr(model_paths[i], ctx);
      dlr_models.push_back(model_ptr);
    } catch (dmlc::Error& e) {
      LOG(ERROR) << e.what();
      return -1;
    }
  }
  DLRModel* pipeline_model = new PipelineModel(dlr_models, ctx);
  *handle = pipeline_model;
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

extern "C" int GetDLRDeviceType(const char* model_path) {
  API_BEGIN();
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

  try {
    return dlr::GetDeviceTypeFromMetadata(path_vec);
  } catch (dmlc::Error& e) {
    LOG(ERROR) << e.what();
    return -1;
  }
  API_END();
}

extern "C" int GetDLRVersion(const char** out) {
  API_BEGIN();
  static const std::string version_str = std::to_string(DLR_MAJOR) + "." +
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
