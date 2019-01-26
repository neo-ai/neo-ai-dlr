#ifndef DLR_H_
#define DLR_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/module.h>
#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
//#include <graph/graph_runtime.cc>
//#include <graph/debug/graph_runtime_debug.cc>
#include <runtime_base.h>
#include <dlpack/dlpack.h>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>

#define LINE_SIZE 256

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

class DLRModel {
 private:
  std::string version_;
  DLRBackend backend_;
  size_t num_inputs_ = 1;
  size_t num_outputs_ = 1;
  /* fields for TVM model */
  std::shared_ptr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::shared_ptr<tvm::runtime::Module> tvm_module_;
  std::vector<const DLTensor *> outputs_;
  std::vector<std::string> input_names_;
  DLContext ctx_;
  /* fields for Treelite model */
  PredictorHandle treelite_model_;
  size_t treelite_num_feature_;
  size_t treelite_output_size_;
  std::unique_ptr<TreelitePredictorEntry[]> treelite_input_;
  std::unique_ptr<float[]> treelite_output_;

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
  void SetInput(const char* name, const int64_t* shape, float* input, int dim);
  void GetNumOutputs(int* num_outputs) const;
  void GetOutputShape(int index, int64_t* shape) const;
  void GetOutputSizeDim(int index, int64_t* size, int* dim);
  void GetOutput(int index, float* out);

  /*! /brief DLRModel destructor
   */
  virtual ~DLRModel() {};

};


/*! /brief Handle for DLRModel
 */
typedef void* DLRModelHandle;

/*! /brief Create a DLR model.
 */
extern "C" int CreateDLRModel(DLRModelHandle *handle,
                              const char *model_path,
                              int dev_type, int dev_id);

/*! /brief Delete a DLR model.
 */
extern "C" int DeleteDLRModel(DLRModelHandle* handle);

/*! /brief Run a DLR model.
 */
extern "C" int RunDLRModel(DLRModelHandle *handle);

/*!
 * \brief Get the number of inputs.
 * \param handle The model handler.
 * \param num_inputs The number of inputs.
 */
extern "C" int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs);

/*!
 * \brief Get name of the index-th input.
 * \param handle The model handler.
 * \param index The index of the input.
 * \param input_name The name of the index-th input.
 */
extern "C" int GetDLRInputName(DLRModelHandle* handle, int index,
                               const char** input_name);

/*!
 * \brief Set the input according the node name.
 * \param handle The model handler.
 * \param name The input node name.
 * \param shape The input node shape.
 * \param input The data for the input.
 * \param dim The dimension of the input data.
 */
extern "C" int SetDLRInput(DLRModelHandle* handle,
                           const char* name,
                           const int64_t* shape,
                           float* input,
                           int dim);
/*!
 * \brief Get the shape of the index-th output.
 * \param handle The model handler.
 * \param index The index-th output.
 * \param shape The shape of index-th output.
 */
extern "C" int GetDLROutputShape(DLRModelHandle* handle,
                                 int index,
                                 int64_t* shape);

/*!
 * \brief Get the index-th output from the model.
 * \param handle The model handler.
 * \param index The index-th output.
 * \param out The pointer to save the output data.
 */
extern "C" int GetDLROutput(DLRModelHandle* handle,
                            int index,
                            float* out);
/*!
 * \brief Get the number of outputs.
 * \param handle The model handler.
 * \param num_outputs The pointer to save the number of outputs.
 */
extern "C" int GetDLRNumOutputs(DLRModelHandle* handle,
                                int* num_outputs);

/*!
 * \brief Get the size and dimension of an output.
 * \param handle The model handler.
 * \param index The index-th output.
 * \param size The size of the index-th output.
 * \param dim The dimension of the index-th output.
 */
extern "C" int GetDLROutputSizeDim(DLRModelHandle* handle, int index,
                                   int64_t* size, int* dim);
/*!
 * \brief Get the last error message
 * \return null-terminated string
 */
extern "C" const char* DLRGetLastError();

} // namespace dlr

#endif
