#ifndef DLR_H_
#define DLR_H_

#include <stdint.h>

/* special symbols for DLL library on Windows */
#ifdef __cplusplus
extern "C" {  // Open extern "C" block
#endif        // __cplusplus

#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL __declspec(dllexport)
#else
#define DLR_DLL
#endif  // defined(_MSC_VER) || defined(_WIN32)

/*! \brief major version */
#define DLR_MAJOR 1
/*! \brief minor version */
#define DLR_MINOR 2
/*! \brief patch version */
#define DLR_PATCH 0
/*! \brief DLR version */
#define DLR_VERSION (DLR_MAJOR * 10000 + DLR_MINOR * 100 + DLR_PATCH)
/*! \brief helper for making version number */
#define DLR_MAKE_VERSION(major, minor, patch) \
  ((major)*10000 + (minor)*100 + patch)

typedef void* DLRModelHandle;

DLR_DLL
int CreateDLRModel(DLRModelHandle* handle, const char* model_path, int dev_type,
                   int dev_id);

DLR_DLL
int DeleteDLRModel(DLRModelHandle* handle);

DLR_DLL
int RunDLRModel(DLRModelHandle* handle);

DLR_DLL
int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs);

DLR_DLL
int GetDLRInputName(DLRModelHandle* handle, int index, const char** input_name);

DLR_DLL
int GetDLRInputType(DLRModelHandle* handle, int index, const char** input_type);

DLR_DLL
int SetDLRInput(DLRModelHandle* handle, const char* name, const int64_t* shape,
                void* input, int dim);
DLR_DLL
int GetDLRInput(DLRModelHandle* handle, const char* name, void* input);

DLR_DLL
int GetDLROutputShape(DLRModelHandle* handle, int index, int64_t* shape);

DLR_DLL
int GetDLROutput(DLRModelHandle* handle, int index, void* out);

DLR_DLL
int GetDLRNumOutputs(DLRModelHandle* handle, int* num_outputs);

DLR_DLL
int GetDLROutputSizeDim(DLRModelHandle* handle, int index, int64_t* size,
                        int* dim);

DLR_DLL
int GetDLROutputType(DLRModelHandle* handle, int index,
                     const char** output_type);

DLR_DLL int GetDLRHasMetadata(DLRModelHandle* handle, bool* has_metadata);

DLR_DLL int GetDLROutputName(DLRModelHandle* handle, const int index,
                             const char** name);

DLR_DLL int GetDLROutputIndex(DLRModelHandle* handle, const char* name,
                              int* index);

DLR_DLL int GetDLROutputByName(DLRModelHandle* handle, const char* name,
                               void* out);
DLR_DLL
const char* DLRGetLastError();

DLR_DLL
int GetDLRBackend(DLRModelHandle* handle, const char** name);

DLR_DLL
int GetDLRVersion(const char** out);

DLR_DLL int SetDLRNumThreads(DLRModelHandle* handle, int threads);

DLR_DLL
int UseDLRCPUAffinity(DLRModelHandle* handle, int use);

/*! \} */

#ifdef __cplusplus
}  // Close extern "C" block
#endif  // __cplusplus

#endif  // DLR_H_
