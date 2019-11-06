#ifndef DLR_TVM_H_
#define DLR_TVM_H_

#include <graph/graph_runtime.h>
#include "dlr_common.h"

namespace dlr {

/*! \brief Get the paths of the TVM model files.
 */
ModelPath GetTvmPaths(std::vector<std::string> tar_path);


/*! \brief class TVMModel
 */
class TVMModel: public DLRModel {
 private:
  std::shared_ptr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::shared_ptr<tvm::runtime::Module> tvm_module_;
  std::vector<const DLTensor *> outputs_;
  std::vector<std::string> weight_names_;
  void SetupTVMModule(std::vector<std::string> model_path);
 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TVMModel(std::vector<std::string> model_path, 
                    const DLContext& ctx): DLRModel(ctx, DLRBackend::kTVM) {
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
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
};

} // namespace dlr


#endif  // DLR_TVM_H_