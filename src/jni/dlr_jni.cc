#include <jni.h>
#include <cstdint>
#include "dlr.h"

// using namespace dlr;

/* DLR C API implementation */
extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRNumInputs(JNIEnv* env, jobject thiz,
                                            jlong jhandle) {
    int num;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLRNumInputs(handle, &num)) {
        return -1;
    }
    return num;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRNumWeights(JNIEnv* env, jobject thiz,
                                             jlong jhandle) {
    int num;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLRNumWeights(handle, &num)) {
        return -1;
    }
    return num;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRInputName(JNIEnv* env, jobject thiz,
                                            jlong jhandle,
                                            jint index) {
    const char* name;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLRInputName(handle, index, &name)) {
        return NULL;
    }
    return env->NewStringUTF(name);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRWeightName(JNIEnv* env, jobject thiz,
                                             jlong jhandle,
                                             jint index) {
    const char* name;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLRWeightName(handle, index, &name)) {
        return NULL;
    }
    return env->NewStringUTF(name);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_SetDLRInput(JNIEnv* env, jobject thiz,
                                        jlong jhandle,
                                        jstring jname,
                                        jlongArray shape,
                                        jfloatArray input,
                                        jint dim) {
    jboolean isCopy = JNI_FALSE;
    jfloat* input_body = env->GetFloatArrayElements(input, &isCopy);
    jlong* shape_body = env->GetLongArrayElements(shape, &isCopy);
    const char* name = env->GetStringUTFChars(jname, &isCopy);
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    int res = SetDLRInput(handle, name, shape_body, input_body, dim);
    env->ReleaseLongArrayElements(shape, shape_body, 0);
    return res;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRInput(JNIEnv* env, jobject thiz,
                                        jlong jhandle,
                                        jstring jname,
                                        jfloatArray input) {
    jboolean isCopy = JNI_FALSE;
    jfloat* arr_body = env->GetFloatArrayElements(input, &isCopy);
    const char* name = env->GetStringUTFChars(jname, &isCopy);
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    int res = GetDLRInput(handle, name, arr_body);
    env->ReleaseFloatArrayElements(input, arr_body, 0);
    return res;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLROutputShape(JNIEnv* env, jobject thiz,
                                              jlong jhandle,
                                              jint index,
                                              jlongArray shape) {
    jboolean isCopy = JNI_FALSE;
    jlong* arr_body = env->GetLongArrayElements(shape, &isCopy);
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    int res = GetDLROutputShape(handle, index, arr_body);
    env->ReleaseLongArrayElements(shape, arr_body, 0);
    return res;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLROutput(JNIEnv* env, jobject thiz,
                                         jlong jhandle,
                                         jint index,
                                         jfloatArray output) {
    jboolean isCopy = JNI_FALSE;
    jfloat* arr_body = env->GetFloatArrayElements(output, &isCopy);
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    int res = GetDLROutput(handle, index, arr_body);
    env->ReleaseFloatArrayElements(output, arr_body, 0);
    return res;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLROutputDim(JNIEnv* env, jobject thiz,
                                            jlong jhandle,
                                            jint index) {
    int64_t out_size;
    int out_dim;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLROutputSizeDim(handle, index, &out_size, &out_dim)) {
        return -1;
    }
    return out_dim;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLROutputSize(JNIEnv* env, jobject thiz,
                                             jlong jhandle,
                                             jint index) {
    int64_t out_size;
    int out_dim;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLROutputSizeDim(handle, index, &out_size, &out_dim)) {
        return -1;
    }
    return out_size;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRNumOutputs(JNIEnv* env, jobject thiz,
                                             jlong jhandle) {
    int num;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLRNumOutputs(handle, &num)) {
        return -1;
    }
    return num;
}

#ifdef DLR_TFLITE
extern "C" JNIEXPORT jlong JNICALL
Java_com_amazon_neo_dlr_DLR_CreateDLRModelFromTFLite(JNIEnv* env, jobject thiz,
                                                     jstring jmodel_path,
                                                     jint threads,
                                                     jint use_nnapi) {
  jboolean isCopy = JNI_FALSE;
  const char* model_path = env->GetStringUTFChars(jmodel_path, &isCopy);
  DLRModelHandle* handle = new DLRModelHandle();
  if (CreateDLRModelFromTFLite(handle, model_path, threads, use_nnapi)) {
      // FAIL
      return 0;
  }
  // Return handle as jlong
  std::uintptr_t jhandle = reinterpret_cast<std::uintptr_t>(handle);
  return jhandle;
}
#endif // DLR_TFLITE

/*! \brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" JNIEXPORT jlong JNICALL
Java_com_amazon_neo_dlr_DLR_CreateDLRModel(JNIEnv* env, jobject thiz,
                                           jstring jmodel_path,
                                           jint dev_type, jint dev_id) {
    jboolean isCopy = JNI_FALSE;
    const char* model_path = env->GetStringUTFChars(jmodel_path, &isCopy);
    DLRModelHandle* handle = new DLRModelHandle();
    if (CreateDLRModel(handle, model_path, dev_type, dev_id)) {
        // FAIL
        return 0;
    }
    // Return handle as jlong
    std::uintptr_t jhandle = reinterpret_cast<std::uintptr_t>(handle);
    return jhandle;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_DeleteDLRModel(JNIEnv* env, jobject thiz,
                                           jlong jhandle) {
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    return DeleteDLRModel(handle);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_RunDLRModel(JNIEnv* env, jobject thiz,
                                        jlong jhandle) {
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    return RunDLRModel(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_amazon_neo_dlr_DLR_DLRGetLastError(JNIEnv* env, jobject thiz) {
    const char* err = DLRGetLastError();
    return env->NewStringUTF(err);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_amazon_neo_dlr_DLR_GetDLRBackend(JNIEnv* env, jobject thiz,
                                          jlong jhandle) {
    const char* name;
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    if (GetDLRBackend(handle, &name)) {
        return NULL;
    }
    return env->NewStringUTF(name);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_SetDLRNumThreads(JNIEnv* env, jobject thiz,
                                             jlong jhandle,
                                             jint threads) {
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    return SetDLRNumThreads(handle, threads);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR_UseDLRCPUAffinity(JNIEnv* env, jobject thiz,
                                             jlong jhandle,
                                             jboolean use) {
    DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
    return UseDLRCPUAffinity(handle, use);
}
