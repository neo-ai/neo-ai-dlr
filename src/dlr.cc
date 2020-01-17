#include "dlr.h"

#include "dlr_common.h"
#include "dlr_tvm.h"
#include "dlr_treelite.h"
#ifdef DLR_TFLITE
#include "dlr_tflite/dlr_tflite.h"
#endif // DLR_TFLITE


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

  std::vector<std::string> path_vec = dmlc::Split(model_path, ':');

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


extern "C" int SetDataCollectionConsent(DLRModelHandle* handle, int flag) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  CallHome(1, "", flag); 
  API_END();
}

#if defined(__ANDROID__)
std::string ext_path;
std::string uuid_;
void get_external_storage_path(JNIEnv* env, jobject instance)
{
  jclass envcls = env->FindClass("android/os/Environment");
  if (envcls == 0) {
    return;
  }
  jmethodID mid = env->GetStaticMethodID(envcls,
                                        "getExternalStorageDirectory", "()Ljava/io/File;");
  if (mid == 0) {
    return;
  }
  jobject fileext = env->CallStaticObjectMethod(envcls, mid);
  if (fileext == 0) {
    return;
  }
  jmethodID midf = env->GetMethodID(env->GetObjectClass(fileext),
                                                       "getAbsolutePath", "()Ljava/lang/String;");
  if (midf == 0) {
    return;
  }
  jstring path = (jstring) env->CallObjectMethod(fileext, midf);
  size_t length = (size_t) env->GetStringLength(path);
  ext_path.assign(env->GetStringUTFChars(path, 0));
}

void get_uuid()
{
  FILE *fp;
  std::string result;
  fp = popen("/system/bin/ip link", "r");
  if (fp == NULL) {
    fp = popen("/usr/bin/ifconfig -a", "r");
    if (fp == NULL)
      LOG(FATAL) << "System command failed to retrieve uuid "; 
    exit(1);
  }
  size_t len;
  ssize_t read;
  char * line = NULL;
  std::string sub_str;
  while((read =getline(&line, &len, fp)) != -1) {
    std::string str_line (line);
    size_t p = str_line.find("link/ether");
    if (p != std::string::npos) {
      sub_str = str_line.substr(str_line.find(' ', p), 18);
      sub_str.erase(std::remove(sub_str.begin(), sub_str.end(), ':'), sub_str.end());
      std::string::size_type sz=0;
      unsigned long long dev_id = stoull(sub_str, &sz, 0);
      if (dev_id > 0) {
        std::size_t uuid_hash = std::hash<std::string>{}(sub_str);
        uuid_.assign(std::to_string(uuid_hash).c_str());
        break;
      }
    }
  }
  pclose(fp);
}
#endif
