#ifndef DLR_H_
#define DLR_H_

#include <stddef.h>
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
#define DLR_MINOR 6
/*! \brief patch version */
#define DLR_PATCH 0
/*! \brief DLR version */
#define DLR_VERSION (DLR_MAJOR * 10000 + DLR_MINOR * 100 + DLR_PATCH)
/*! \brief helper for making version number */
#define DLR_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + patch)

/*!
 * \defgroup c_api
 * C API of DLR
 * \{
 */

/*!
 \brief Handle for DLRModel.
 */
typedef void* DLRModelHandle;

#ifndef DLR_ALLOC_TYPEDEF
#define DLR_ALLOC_TYPEDEF
/*! \brief A pointer to a malloc-like function. */
typedef void* (*DLRMallocFunctionPtr)(size_t);
/*! \brief A pointer to a free-like function. */
typedef void (*DLRFreeFunctionPtr)(void*);
/*! \brief A pointer to a memalign-like function. */
typedef void* (*DLRMemalignFunctionPtr)(size_t, size_t);
#endif

#ifndef DLR_MODEL_ELEM
#define DLR_MODEL_ELEM
enum DLRModelElemType { HEXAGON_LIB, NEO_METADATA, TVM_GRAPH, TVM_LIB, TVM_PARAMS, RELAY_EXEC };
typedef struct ModelElem {
  const DLRModelElemType type;
  const char* path;
  const void* data;
  const size_t data_size;
} DLRModelElem;
#endif

/*!
 * \brief Creates a DLR model
 * \param handle The pointer to save the model handle.
 * \param model_path Path to the folder containing the model files,
 *                   or colon-separated list of folders containing model files,
 *                   or colon-separated list of paths to model files
 * \param dev_type Device type. Valid values are in the DLDeviceType enum in dlpack.h.
 * \param dev_id Device ID.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int CreateDLRModel(DLRModelHandle* handle, const char* model_path, int dev_type, int dev_id);

/*!
 \brief Creates a DLR model from model elements.
 \param handle The pointer to save the model handle.
 \param model_elems DLR Model elements. Element can be file path or data pointer in memory.
 \param dev_type Device type. Valid values are in the DLDeviceType enum in dlpack.h.
 \param dev_id Device ID.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int CreateDLRModelFromModelElem(DLRModelHandle* handle, const DLRModelElem* model_elems,
                                size_t model_elems_size, int dev_type, int dev_id);

#ifdef DLR_HEXAGON
/*!
 \brief Creates a DLR model from Hexagon
 \param handle The pointer to save the model handle.
 \param model_path Path to _hexagon_model.so file or to the top-level directory containing the file
 \param debug_level 0 - no debug, 100 - max debug.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int CreateDLRModelFromHexagon(DLRModelHandle* handle, const char* model_path, int debug_level);
#endif  // DLR_HEXAGON

/*!
 \brief Creates a DLR pipeline model.
 \param handle The pointer to save the model handle.
 \param num_models Number of items in model_paths array
 \param model_paths Paths to the folders containing the models files,
                    or colon-separated list of folders (or files) if model files
                    stored in different locations
 \param dev_type Device type. Valid values are in the DLDeviceType enum in dlpack.h.
 \param dev_id Device ID.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int CreateDLRPipeline(DLRModelHandle* handle, int num_models, const char** model_paths,
                      int dev_type, int dev_id);

/*!
 \brief Deletes a DLR model.
 \param handle The model handle returned from CreateDLRModel().
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int DeleteDLRModel(DLRModelHandle* handle);

/*!
 \brief Runs a DLR model.
 \param handle The model handle returned from CreateDLRModel().
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int RunDLRModel(DLRModelHandle* handle);

/*!
 \brief Gets the number of inputs.
 \param handle The model handle returned from CreateDLRModel().
 \param num_inputs The pointer to save the number of inputs.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs);

/*!
 \brief Gets the number of weights.
 \param handle The model handle returned from CreateDLRModel().
 \param num_weights The pointer to save the number of weights.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRNumWeights(DLRModelHandle* handle, int* num_weights);

/*!
 \brief Gets the name of the index-th input.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index of the input.
 \param input_name The pointer to save the name of the index-th input.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRInputName(DLRModelHandle* handle, int index, const char** input_name);

/*!
 \brief Gets the type of the index-th input.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index of the input.
 \param input_type The pointer to save the type of the index-th input.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRInputType(DLRModelHandle* handle, int index, const char** input_type);

/*!
 \brief Gets the name of the index-th weight.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index of the weight.
 \param input_name The pointer to save the name of the index-th weight.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRWeightName(DLRModelHandle* handle, int index, const char** weight_name);

/*!
 \brief Sets the input according the node name.
 \param handle The model handle returned from CreateDLRModel().
 \param name The input node name.
 \param shape The input node shape as an array.
 \param input The data for the input as an array.
 \param dim The dimension of the input data.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int SetDLRInput(DLRModelHandle* handle, const char* name, const int64_t* shape, const void* input,
                int dim);

/*!
 * \brief Sets the input according the node name from existing DLTensor. Can only be
 *        used with TVM models (GraphRuntime and VMRuntime)
 * \param handle The model handle returned from CreateDLRModel().
 * \param name The input node name.
 * \param tensor The input DLTensor.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int SetDLRInputTensor(DLRModelHandle* handle, const char* name, void* tensor);

/*!
 \brief Gets the current value of the input according the node name.
 \param handle The model handle returned from CreateDLRModel().
 \param name The input node name.
 \param input The current value of the input will be copied to this buffer.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRInput(DLRModelHandle* handle, const char* name, void* input);

/*!
 \brief Gets the shape of the index-th input.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th input.
 \param shape The pointer to save the shape of index-th input. This should be a
 pointer to an array of size "dim" from GetDLRInputSizeDim().
 \return 0 for
 success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int GetDLRInputShape(DLRModelHandle* handle, int index, int64_t* shape);

/*!
 \brief Gets the size and dimension of an input.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th input.
 \param size The pointer to save the size of the index-th input.
 \param dim The pointer to save the dimension of the index-th output.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRInputSizeDim(DLRModelHandle* handle, int index, int64_t* size, int* dim);

/*!
 \brief Gets the shape of the index-th output.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param shape The pointer to save the shape of index-th output. This should be a
 pointer to an array of size "dim" from GetDLROutputSizeDim(). \return 0 for
 success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int GetDLROutputShape(DLRModelHandle* handle, int index, int64_t* shape);

/*!
 \brief Gets the index-th output from the model.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param out The pointer to save the output data. This should be a pointer to an
 array of size "size" from GetDLROutputSizeDim(). \return 0 for success, -1 for
 error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int GetDLROutput(DLRModelHandle* handle, int index, void* out);

/*!
 \brief Gets the index-th output from the model.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param out Storage to save output pointer
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int GetDLROutputPtr(DLRModelHandle* handle, int index, const void** out);

/*!
 * \brief Gets the index-th output from the model and copies it into the given DLTensor.
 *        Can only be used with TVM models (GraphRuntime and VMRuntime)
 * \param handle The model handle returned from CreateDLRModel().
 * \param index The index-th output.
 * \param tensor The pointer to an existing/allocated DLTensor to copy the output into.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int GetDLROutputTensor(DLRModelHandle* handle, int index, void* tensor);

/*!
 * \brief Gets the index-th output from the model and sets the pointer to it.
 *        Can only be used with TVM models (GraphRuntime and VMRuntime)
 * \param handle The model handle returned from CreateDLRModel().
 * \param index The index-th output.
 * \param tensor The pointer to an unallocated DLManagedTensor pointer, will be
 *               set by this function to point to an internal DLManagedTensor.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int GetDLROutputManagedTensorPtr(DLRModelHandle* handle, int index, const void** tensor);

/*!
 \brief Gets the number of outputs.
 \param handle The model handle returned from CreateDLRModel().
 \param num_outputs The pointer to save the number of outputs.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRNumOutputs(DLRModelHandle* handle, int* num_outputs);

/*!
 \brief Gets the size and dimension of an output.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param size The pointer to save the size of the index-th output.
 \param dim The pointer to save the dimension of the index-th output.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLROutputSizeDim(DLRModelHandle* handle, int index, int64_t* size, int* dim);

/*!
 \brief Gets the type of the index-th output.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index of the output.
 \param output_type The pointer to save the type of the index-th output.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLROutputType(DLRModelHandle* handle, int index, const char** output_type);

/*!
 \brief Check if metadata file is found in the compilation artifact
 \param handle The model handle returned from CreateDLRModel().
 \param has_metadata The pointer to save boolean value to indicate the presence of metadata file.
*/
DLR_DLL int GetDLRHasMetadata(DLRModelHandle* handle, bool* has_metadata);

/*!
 \brief Gets the output node names of the uncompiled model from the metadata file
  \param handle The model handle returned from CreateDLRModel().
  \param names The pointer to save array containing output node names.
*/
DLR_DLL int GetDLROutputName(DLRModelHandle* handle, const int index, const char** name);

/*!
 \brief Gets the output node index using the node name
  \param handle The model handle returned from CreateDLRModel().
  \param name The pointer pointing to the output node name.
  \param index The pointer to save the corresponding index of the output node.
*/
DLR_DLL int GetDLROutputIndex(DLRModelHandle* handle, const char* name, int* index);

/*!
 \brief Gets the output of the node of the given name from the model.
 \param handle The model handle returned from CreateDLRModel().
 \param name The name of the output node.
 \param out The pointer to save the output data. This should be a pointer to an
 array of size "size" from GetDLROutputSizeDim(). \return 0 for success, -1 for
 error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLROutputByName(DLRModelHandle* handle, const char* name, void* out);

/*!
 \brief Gets the last error message.
 \return Null-terminated string containing the error message.
 */
DLR_DLL
const char* DLRGetLastError();

/*!
 \brief Gets the name of the backend ("tvm", "treelite" or "tflite")
 \param handle The model handle returned from CreateDLRModel().
 \param name The pointer to save the null-terminated string containing the name.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRBackend(DLRModelHandle* handle, const char** name);

/*!
 \brief Gets the DLDeviceType (DLDeviceType::kDLCPU, DLDeviceType::kDLGPU, etc)
 \param model_path Path to the folder containing the model files,
                   or colon-separated list of folders (or files) if model files
 \return DLDeviceType enum for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int GetDLRDeviceType(const char* model_path);

/*!
 \brief Get DLR version
 \param out The pointer to save the null-terminated string containing the
 version. \return 0 for success, -1 for error. Call DLRGetLastError() to get the
 error message.
 */
DLR_DLL
int GetDLRVersion(const char** out);

/*!
 \brief Set the number of threads available to DLR
 \param handle The model handle returned from CreateDLRModel().
 \param threads number of threads
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int SetDLRNumThreads(DLRModelHandle* handle, int threads);

/*!
 \brief Enable or disable CPU Affinity
 \param handle The model handle returned from CreateDLRModel().
 \param use 0 to disable, 1 to enable
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error
 message.
 */
DLR_DLL
int UseDLRCPUAffinity(DLRModelHandle* handle, int use);

/*!
 * \brief Set custom allocator malloc function. Must be called before CreateDLRModel or
 *        CreateDLRPipeline. It is recommended to use with SetDLRCustomAllocatorFree and
 *        SetDLRCustomAllocatorMemalign.
 * \param custom_memalign_fn Function pointer to memalign-like function.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int SetDLRCustomAllocatorMalloc(DLRMallocFunctionPtr custom_malloc_fn);

/*!
 * \brief Set custom allocator free function. Must be called before CreateDLRModel or
 *        CreateDLRPipeline. It is recommended to use with SetDLRCustomAllocatorMalloc and
 *        SetDLRCustomAllocatorMemalign.
 * \param custom_free_fn Function pointer to free-like function.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int SetDLRCustomAllocatorFree(DLRFreeFunctionPtr custom_free_fn);

/*!
 * \brief Set custom allocator memalign function. memalign is used heavily by the TVM and RelayVM
 *        backends. Must be called before CreateDLRModel or CreateDLRPipeline. It is recommended
 *        to use with SetDLRCustomAllocatorMalloc and  SetDLRCustomAllocatorFree.
 * \param custom_memalign_fn Function pointer to memalign-like function.
 * \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL
int SetDLRCustomAllocatorMemalign(DLRMemalignFunctionPtr custom_memalign_fn);

/*! \} */

#ifdef __cplusplus
}  // Close extern "C" block
#endif  // __cplusplus

#endif  // DLR_H_
