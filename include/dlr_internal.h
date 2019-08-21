#ifndef DLR_INTERNAL_H_
#define DLR_INTERNAL_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/module.h>
#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
#include "../3rdparty/treelite/dmlc-core/src/io/filesys.h"
#include <runtime_base.h>
#include <dlpack/dlpack.h>
#include <string>
#include <vector>
#include <sys/types.h>
#include "dlr.h"

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
  /*! /brief Load model files from given folder path.
   */
  explicit DLRModel(const std::string& model_path,
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


#endif  // DLR_INTERNAL_H_
