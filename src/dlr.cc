#include "dlr.h"

#include "dlr_common.h"
#include "dlr_tvm.h"
#include "dlr_treelite.h"
#ifdef DLR_TFLITE
#include "dlr_tflite/dlr_tflite.h"
#endif // DLR_TFLITE

#include <locale>


using namespace dlr;

/* DLR C API implementation */

extern "C" int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetNumInputs(num_inputs);
  API_END();
}

extern "C" int GetDLRNumWeights(DLRModelHandle* handle, int* num_weights) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetNumWeights(num_weights);
  API_END();
}

extern "C" int GetDLRInputName(DLRModelHandle* handle, int index,
                               const char** input_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *input_name = model->GetInputName(index);
  API_END();
}

extern "C" int GetDLRWeightName(DLRModelHandle* handle, int index,
                                const char** weight_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *weight_name = model->GetWeightName(index);
  API_END();
}

extern "C" int SetDLRInput(DLRModelHandle* handle,
                           const char* name,
                           const int64_t* shape,
                           float* input,
                           int dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetInput(name, shape, input, dim);
  API_END();
}

extern "C" int GetDLRInput(DLRModelHandle* handle,
                           const char* name,
                           float* input) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetInput(name, input);
  API_END();
}

extern "C" int GetDLROutputShape(DLRModelHandle* handle,
                                 int index,
                                 int64_t* shape) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputShape(index, shape);
  API_END();
}

extern "C" int GetDLROutput(DLRModelHandle* handle,
                            int index,
                            float* out) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutput(index, out);
  API_END();
}

extern "C" int GetDLROutputSizeDim(DLRModelHandle* handle,
                                   int index,
                                   int64_t* size,
                                   int* dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputSizeDim(index, size, dim);
  API_END();
}

extern "C" int GetDLRNumOutputs(DLRModelHandle* handle,
                                int* num_outputs) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetNumOutputs(num_outputs);
  API_END();
}

#ifdef DLR_TFLITE
/*! \brief Translate c args from ctypes to std types for DLRModelFromTFLite ctor.
 */
int CreateDLRModelFromTFLite(DLRModelHandle *handle,
                   const char *model_path,
                   int threads,
                   int use_nnapi) {
  API_BEGIN();
  const std::string model_path_string(model_path);
  // TFLiteModel class does not use DLContext internally
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(1); // 1 - kDLCPU
  ctx.device_id = 0;
  DLRModel* model = new TFLiteModel(model_path_string, ctx, threads, (bool) use_nnapi);
  *handle = model;
  API_END();
}
#endif // DLR_TFLITE

/*! \brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" int CreateDLRModel(DLRModelHandle* handle,
                                const char* model_path,
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
#ifdef DLR_TFLITE
  } else if (backend == DLRBackend::kTFLITE) {
    // By default use undefined number of threads - threads=0 and use_nnapi=0
    DLRModelHandle tf_handle;
    CreateDLRModelFromTFLite(&tf_handle, model_path, 0, 0);
    model = static_cast<DLRModel *>(tf_handle);
#endif // DLR_TFLITE
  } else {
    LOG(FATAL) << "Unsupported backend!";
    return -1; // unreachable
  }
  *handle = model;
  API_END();
}

extern "C" int DeleteDLRModel(DLRModelHandle* handle) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  delete model;
  API_END();
}

extern "C" int RunDLRModel(DLRModelHandle *handle) {
  API_BEGIN();
  static_cast<DLRModel *>(*handle)->Run();
  API_END();
}

extern "C" const char* DLRGetLastError() {
  return TVMGetLastError();
}

extern "C" int GetDLRBackend(DLRModelHandle* handle, const char** name) {
  API_BEGIN();
  *name = static_cast<DLRModel *>(*handle)->GetBackend();
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
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetNumThreads(threads);
  API_END();
}

extern "C" int UseDLRCPUAffinity(DLRModelHandle* handle, int use) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->UseCPUAffinity(use);
  API_END();
}
