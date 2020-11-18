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

extern "C" int GetDLRInputName(DLRModelHandle* handle, int index, const char** input_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *input_name = model->GetInputName(index);
  API_END();
}

extern "C" int GetDLRInputType(DLRModelHandle* handle, int index, const char** input_type) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *input_type = model->GetInputType(index);
  API_END();
}

extern "C" int GetDLRInputShape(DLRModelHandle* handle, int index, int64_t* shape) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  const std::vector<int64_t>& input_shape = model->GetInputShape(index);
  std::copy(input_shape.begin(), input_shape.end(), shape);
  API_END();
}

extern "C" int GetDLRInputSizeDim(DLRModelHandle* handle, int index, int64_t* size, int* dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *size = model->GetInputSize(index);
  *dim = model->GetInputDim(index);
  API_END();
}

extern "C" int GetDLRWeightName(DLRModelHandle* handle, int index, const char** weight_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *weight_name = model->GetWeightName(index);
  API_END();
}

extern "C" int SetDLRInput(DLRModelHandle* handle, const char* name, const int64_t* shape,
                           const void* input, int dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetInput(name, shape, input, dim);
  API_END();
}

extern "C" int SetDLRInputTensor(DLRModelHandle* handle, const char* name, void* tensor) {
  API_BEGIN();
  DLRModel* dlr_model = static_cast<DLRModel*>(*handle);
  DLRBackend backend = dlr_model->GetBackend();
  CHECK(backend == DLRBackend::kTVM || backend == DLRBackend::kRELAYVM)
      << "model is not a TVMModel or RelayVMModel. Found '"
      << kBackendToStr[static_cast<int>(backend)] << "' but expected 'tvm' or 'relayvm'";

  DLTensor* dltensor = static_cast<DLTensor*>(tensor);
  if (backend == DLRBackend::kTVM) {
    TVMModel* tvm_model = static_cast<TVMModel*>(*handle);
    std::cout << "here" << std::endl;
    CHECK(tvm_model != nullptr) << "model is nullptr, create it first";
    tvm_model->SetInputTensor(name, dltensor);
  } else {
    RelayVMModel* vm_model = static_cast<RelayVMModel*>(*handle);
    CHECK(vm_model != nullptr) << "model is nullptr, create it first";
    vm_model->SetInputTensor(name, dltensor);
  }
  API_END();
}

extern "C" int GetDLRInput(DLRModelHandle* handle, const char* name, void* input) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetInput(name, input);
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

extern "C" int GetDLROutputTensor(DLRModelHandle* handle, int index, void* tensor) {
  API_BEGIN();
  DLRModel* dlr_model = static_cast<DLRModel*>(*handle);
  DLRBackend backend = dlr_model->GetBackend();
  CHECK(backend == DLRBackend::kTVM || backend == DLRBackend::kRELAYVM)
      << "model is not a TVMModel or RelayVMModel. Found '"
      << kBackendToStr[static_cast<int>(backend)] << "' but expected 'tvm' or 'relayvm'";

  DLTensor* dltensor = static_cast<DLTensor*>(tensor);
  if (backend == DLRBackend::kTVM) {
    TVMModel* tvm_model = static_cast<TVMModel*>(*handle);
    CHECK(tvm_model != nullptr) << "model is nullptr, create it first";
    tvm_model->GetOutputTensor(index, dltensor);
  } else {
    RelayVMModel* vm_model = static_cast<RelayVMModel*>(*handle);
    CHECK(vm_model != nullptr) << "model is nullptr, create it first";
    vm_model->GetOutputTensor(index, dltensor);
  }
  API_END();
}

extern "C" int GetDLROutputManagedTensorPtr(DLRModelHandle* handle, int index,
                                            const void** tensor) {
  API_BEGIN();
  DLRModel* dlr_model = static_cast<DLRModel*>(*handle);
  DLRBackend backend = dlr_model->GetBackend();
  CHECK(backend == DLRBackend::kTVM || backend == DLRBackend::kRELAYVM)
      << "model is not a TVMModel or RelayVMModel. Found '"
      << kBackendToStr[static_cast<int>(backend)] << "' but expected 'tvm' or 'relayvm'";

  const DLManagedTensor** dltensor = reinterpret_cast<const DLManagedTensor**>(tensor);
  if (backend == DLRBackend::kTVM) {
    TVMModel* tvm_model = static_cast<TVMModel*>(*handle);
    CHECK(tvm_model != nullptr) << "model is nullptr, create it first";
    tvm_model->GetOutputManagedTensorPtr(index, dltensor);
  } else {
    RelayVMModel* vm_model = static_cast<RelayVMModel*>(*handle);
    CHECK(vm_model != nullptr) << "model is nullptr, create it first";
    vm_model->GetOutputManagedTensorPtr(index, dltensor);
  }
  API_END();
}

extern "C" int GetDLROutputSizeDim(DLRModelHandle* handle, int index, int64_t* size, int* dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputSizeDim(index, size, dim);
  API_END();
}

extern "C" int GetDLROutputShape(DLRModelHandle* handle, int index, int64_t* shape) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel*>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputShape(index, shape);
  API_END();
}

extern "C" int GetDLROutputType(DLRModelHandle* handle, int index, const char** output_type) {
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

std::vector<std::string> MakePathVec(std::string model_path) {
  std::vector<std::string> path_vec = dmlc::Split(model_path, ':');
  for (int i = 0; i < path_vec.size(); i++) path_vec[i] = FixWindowsDriveLetter(path_vec[i]);
  return path_vec;
}

DLRModelPtr CreateDLRModelPtr(const char* model_path, DLContext& ctx) {
  std::vector<std::string> path_vec = MakePathVec(model_path);
  std::vector<std::string> files = FindFiles(path_vec);
  DLRBackend backend = dlr::GetBackend(files);
  if (backend == DLRBackend::kTVM) {
    return std::make_shared<TVMModel>(files, ctx);
  } else if (backend == DLRBackend::kRELAYVM) {
    return std::make_shared<RelayVMModel>(files, ctx);
  } else if (backend == DLRBackend::kTREELITE) {
    return std::make_shared<TreeliteModel>(files, ctx);
#ifdef DLR_HEXAGON
  } else if (backend == DLRBackend::kHEXAGON) {
    return std::make_shared<HexagonModel>(files, ctx, 1 /*debug_level*/);
#endif  // DLR_HEXAGON
  } else {
    std::string err = "Unable to determine backend from path: '";
    err = err + model_path + "'.";
    throw dmlc::Error(err);
    return nullptr;  // unreachable
  }
}

#ifdef DLR_HEXAGON
/*! \brief Translate c args from ctypes to std types for DLRModelFromHexagon
 * ctor.
 */
int CreateDLRModelFromHexagon(DLRModelHandle* handle, const char* model_path, int debug_level) {
  API_BEGIN();
  // HexagonModel class does not use DLContext internally
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(1);  // 1 - kDLCPU
  ctx.device_id = 0;
  std::vector<std::string> path_vec = MakePathVec(model_path);
  std::vector<std::string> files = FindFiles(path_vec);
  DLRModel* model = new HexagonModel(files, ctx, debug_level);
  *handle = model;
  API_END();
}
#endif  // DLR_HEXAGON

/*! \brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" int CreateDLRModel(DLRModelHandle* handle, const char* model_path, int dev_type,
                              int dev_id) {
  API_BEGIN();
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = dev_id;

  std::vector<std::string> path_vec = MakePathVec(model_path);
  std::vector<std::string> files = FindFiles(path_vec);

  DLRBackend backend = dlr::GetBackend(files);
  DLRModel* model;
  try {
    if (backend == DLRBackend::kTVM) {
      model = new TVMModel(files, ctx);
    } else if (backend == DLRBackend::kRELAYVM) {
      model = new RelayVMModel(files, ctx);
    } else if (backend == DLRBackend::kTREELITE) {
      model = new TreeliteModel(files, ctx);
#ifdef DLR_HEXAGON
    } else if (backend == DLRBackend::kHEXAGON) {
      model = new HexagonModel(files, ctx, 1 /*debug_level*/);
#endif  // DLR_HEXAGON
    } else {
      std::string err = "Unable to determine backend from path: '";
      err = err + model_path + "'.";
      throw dmlc::Error(err);
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
extern "C" int CreateDLRPipeline(DLRModelHandle* handle, int num_models, const char** model_paths,
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
  DLRBackend backend = static_cast<DLRModel*>(*handle)->GetBackend();
  *name = kBackendToStr[static_cast<int>(backend)];
  API_END();
}

extern "C" int GetDLRDeviceType(const char* model_path) {
  API_BEGIN();
  std::vector<std::string> path_vec = MakePathVec(model_path);
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
  static const std::string version_str =
      std::to_string(DLR_MAJOR) + "." + std::to_string(DLR_MINOR) + "." + std::to_string(DLR_PATCH);
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

extern "C" int SetDLRCustomAllocatorMalloc(DLRMallocFunctionPtr custom_malloc_fn) {
  API_BEGIN();
  DLRAllocatorFunctions::SetMallocFunction(custom_malloc_fn);
  API_END();
}

extern "C" int SetDLRCustomAllocatorFree(DLRFreeFunctionPtr custom_free_fn) {
  API_BEGIN();
  DLRAllocatorFunctions::SetFreeFunction(custom_free_fn);
  API_END();
}

extern "C" int SetDLRCustomAllocatorMemalign(DLRMemalignFunctionPtr custom_memalign_fn) {
  API_BEGIN();
  DLRAllocatorFunctions::SetMemalignFunction(custom_memalign_fn);
  API_END();
}
