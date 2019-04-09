#ifndef DLR_H_
#define DLR_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/module.h>
#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
#include "../3rdparty/treelite/dmlc-core/src/io/filesys.h"
//#include <graph/graph_runtime.cc>
//#include <graph/debug/graph_runtime_debug.cc>
#include <runtime_base.h>
#include <dlpack/dlpack.h>
#include <string>
#include <vector>
#include <sys/types.h>

#define LINE_SIZE 256

/* special symbols for DLL library on Windows */
#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL extern "C" __declspec(dllexport)
#else
#define DLR_DLL extern "C"
#endif

typedef struct {
  std::string model_lib;
  std::string params;
  std::string model_json;
  std::string ver_json;
} ModelPath;

void listdir(const std::string& dirname, std::vector<std::string> &paths);

inline bool endsWith(const std::string &mainStr, const std::string &toMatch)
{
	if(mainStr.size() >= toMatch.size() &&
	  mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
		return true;
	else
		return false;
}



enum class DLRBackend {
  kTVM,
  kTREELITE
};  
/*! /brief Get the paths of the TVM model files.
 */
ModelPath get_tvm_paths(const std::string &tar_path);
 
/*! /brief Get the paths of the Treelite model files.
 */
ModelPath get_treelite_paths(const std::string& dirname);

/*! /brief Get the backend based on the contents of the model folder.
 */
DLRBackend get_backend(const std::string &dirname);

namespace dlr {

struct TreeliteInput {
  std::vector<float> data;
  std::vector<uint32_t> col_ind;
  std::vector<size_t> row_ptr;
  size_t num_row;
  size_t num_col;
  CSRBatchHandle handle;
};

class DLRModel {
 private:
  std::string version_;
  DLRBackend backend_;
  size_t num_inputs_ = 1;
  size_t num_weights_ = 0;
  size_t num_outputs_ = 1;
  /* fields for TVM model */
  std::shared_ptr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::shared_ptr<tvm::runtime::Module> tvm_module_;
  std::vector<const DLTensor *> outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> weight_names_;
  DLContext ctx_;
  /* fields for Treelite model */
  PredictorHandle treelite_model_;
  size_t treelite_num_feature_;
  size_t treelite_output_buffer_size_;  // size of temporary buffer per instance
  size_t treelite_output_size_;  // size of output per instance
  std::unique_ptr<TreeliteInput> treelite_input_;
  std::vector<float> treelite_output_;

 public:
  /*! /brief Extract the .tar file and load the model.
   */
  explicit DLRModel(const std::string& tar_path,
                    const DLContext& ctx);

  /*! /brief Get the output of the given input x.
   */
  void Run();

  void SetupTVMModule(const std::string& model_path);
  void SetupTreeliteModule(const std::string& model_path);
  std::vector<std::string> GetWeightNames() const;
  void GetNumInputs(int* num_inputs) const;
  const char* GetInputName(int index) const;
  const char* GetWeightName(int index) const;
  void SetInput(const char* name, const int64_t* shape, float* input, int dim);
  void GetInput(const char* name, float* input);
  void GetNumOutputs(int* num_outputs) const;
  void GetNumWeights(int* num_weights) const;
  void GetOutputShape(int index, int64_t* shape) const;
  void GetOutputSizeDim(int index, int64_t* size, int* dim);
  void GetOutput(int index, float* out);

  const char* GetBackend() const;

  /*! /brief DLRModel destructor
   */
  virtual ~DLRModel() {};

};

} // namespace dlr

/*!
 * \defgroup c_api
 * C API of DLR
 * \{
 */

/*!
 \brief Handle for DLRModel.
 */
typedef void* DLRModelHandle;

/*!
 \brief Creates a DLR model.
 \param handle The pointer to save the model handle.
 \param model_path Path to the top-level directory containing the model.
 \param dev_type Device type. Valid values are in the DLDeviceType enum in dlpack.h.
 \param dev_id Device ID.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int CreateDLRModel(DLRModelHandle *handle,
                           const char *model_path,
                           int dev_type, int dev_id);

/*!
 \brief Deletes a DLR model.
 \param handle The model handle returned from CreateDLRModel().
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int DeleteDLRModel(DLRModelHandle* handle);

/*!
 \brief Runs a DLR model.
 \param handle The model handle returned from CreateDLRModel().
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int RunDLRModel(DLRModelHandle *handle);

/*!
 \brief Gets the number of inputs.
 \param handle The model handle returned from CreateDLRModel().
 \param num_inputs The pointer to save the number of inputs.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs);

/*!
 \brief Gets the number of weights.
 \param handle The model handle returned from CreateDLRModel().
 \param num_weights The pointer to save the number of weights.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRNumWeights(DLRModelHandle* handle, int* num_weights);

/*!
 \brief Gets the name of the index-th input.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index of the input.
 \param input_name The pointer to save the name of the index-th input.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRInputName(DLRModelHandle* handle, int index,
                            const char** input_name);

/*!
 \brief Gets the name of the index-th weight.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index of the weight.
 \param input_name The pointer to save the name of the index-th weight.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRWeightName(DLRModelHandle* handle, int index,
                             const char** weight_name);

/*!
 \brief Sets the input according the node name.
 \param handle The model handle returned from CreateDLRModel().
 \param name The input node name.
 \param shape The input node shape as an array.
 \param input The data for the input as an array.
 \param dim The dimension of the input data.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int SetDLRInput(DLRModelHandle* handle,
                        const char* name,
                        const int64_t* shape,
                        float* input,
                        int dim);
/*!
 \brief Gets the current value of the input according the node name.
 \param handle The model handle returned from CreateDLRModel().
 \param name The input node name.
 \param input The current value of the input will be copied to this buffer.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRInput(DLRModelHandle* handle,
                        const char* name,
                        float* input);
/*!
 \brief Gets the shape of the index-th output.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param shape The pointer to save the shape of index-th output. This should be a pointer to an array of size "dim" from GetDLROutputSizeDim().
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLROutputShape(DLRModelHandle* handle,
                              int index,
                              int64_t* shape);

/*!
 \brief Gets the index-th output from the model.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param out The pointer to save the output data. This should be a pointer to an array of size "size" from GetDLROutputSizeDim().
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLROutput(DLRModelHandle* handle,
                         int index,
                         float* out);
/*!
 \brief Gets the number of outputs.
 \param handle The model handle returned from CreateDLRModel().
 \param num_outputs The pointer to save the number of outputs.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRNumOutputs(DLRModelHandle* handle,
                             int* num_outputs);

/*!
 \brief Gets the size and dimension of an output.
 \param handle The model handle returned from CreateDLRModel().
 \param index The index-th output.
 \param size The pointer to save the size of the index-th output.
 \param dim The pointer to save the dimension of the index-th output.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLROutputSizeDim(DLRModelHandle* handle, int index,
                                int64_t* size, int* dim);
/*!
 \brief Gets the last error message.
 \return Null-terminated string containing the error message.
 */
DLR_DLL const char* DLRGetLastError();

/*!
 \brief Gets the name of the backend ("tvm" / "treelite")
 \param handle The model handle returned from CreateDLRModel().
 \param name The pointer to save the null-terminated string containing the name.
 \return 0 for success, -1 for error. Call DLRGetLastError() to get the error message.
 */
DLR_DLL int GetDLRBackend(DLRModelHandle* handle, const char** name);

/*! \} */

#endif  // DLR_H_
