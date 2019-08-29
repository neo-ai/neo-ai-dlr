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

// Abstract class
class DLRModel {
 protected:
  std::string version_;
  DLRBackend backend_;
  size_t num_inputs_ = 1;
  size_t num_weights_ = 0;
  size_t num_outputs_ = 1;
  DLContext ctx_;
  std::vector<std::string> input_names_;
 public:
  DLRModel(const DLContext& ctx, const DLRBackend& backend): ctx_(ctx), backend_(backend) {}
  virtual ~DLRModel() {}
  virtual void GetNumInputs(int* num_inputs) const {*num_inputs = num_inputs_;}
  virtual void GetNumOutputs(int* num_outputs) const {*num_outputs = num_outputs_;}
  virtual void GetNumWeights(int* num_weights) const {*num_weights = num_weights_;}
  virtual const char* GetInputName(int index) const =0;
  virtual const char* GetWeightName(int index) const =0;
  virtual std::vector<std::string> GetWeightNames() const =0;
  virtual void GetInput(const char* name, float* input) =0;
  virtual void SetInput(const char* name, const int64_t* shape, float* input, int dim) =0;
  virtual void Run() =0;
  virtual void GetOutput(int index, float* out) =0;
  virtual void GetOutputShape(int index, int64_t* shape) const =0;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) =0;
  virtual const char* GetBackend() const =0;
};


class TVMModel: public DLRModel {
 private:
  std::shared_ptr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::shared_ptr<tvm::runtime::Module> tvm_module_;
  std::vector<const DLTensor *> outputs_;
  std::vector<std::string> weight_names_;
  void SetupTVMModule(const std::string& model_path);
 public:
  /*! /brief Load model files from given folder path.
   */
  explicit TVMModel(const std::string& model_path, const DLContext& ctx): DLRModel(ctx, DLRBackend::kTVM) {
    SetupTVMModule(model_path);
  }

  virtual const char* GetInputName(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, float* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, float* input, int dim) override;
  virtual void Run() override;
  virtual void GetOutput(int index, float* out) override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetBackend() const override;
};


class TreeliteModel: public DLRModel {
 private:
  // fields for Treelite model
  PredictorHandle treelite_model_;
  size_t treelite_num_feature_;
  // size of temporary buffer per instance
  size_t treelite_output_buffer_size_;
  // size of output per instance
  size_t treelite_output_size_;
  std::unique_ptr<TreeliteInput> treelite_input_;
  std::vector<float> treelite_output_;
  void SetupTreeliteModule(const std::string& model_path);
 public:
  /*! /brief Load model files from given folder path.
   */
  explicit TreeliteModel(const std::string& model_path, const DLContext& ctx): DLRModel(ctx, DLRBackend::kTREELITE) {
    SetupTreeliteModule(model_path);
  }

  virtual const char* GetInputName(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, float* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, float* input, int dim) override;
  virtual void Run() override;
  virtual void GetOutput(int index, float* out) override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetBackend() const override;
};

} // namespace dlr


#endif  // DLR_INTERNAL_H_
