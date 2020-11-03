#ifndef DLR_TVM_H_
#define DLR_TVM_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/memory.h>
#include "dlr_common.h"

#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL __declspec(dllexport)
#else
#define DLR_DLL
#endif  // defined(_MSC_VER) || defined(_WIN32)

namespace dlr {

/*! \brief Get the paths of the TVM model files.
 */
ModelPath GetTvmPaths(std::vector<std::string> tar_path);

/*! \brief class TVMModel
 */
class DLR_DLL TVMModel : public DLRModel {
 private:
  tvm::runtime::ObjectPtr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::shared_ptr<tvm::runtime::Module> tvm_module_;
  std::vector<const DLTensor*> outputs_;
  std::vector<std::string> output_types_;
  std::vector<std::string> weight_names_;
  void SetupTVMModule(std::vector<std::string> model_path);
  void UpdateInputShapes();

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TVMModel(std::vector<std::string> model_path, const DLContext& ctx)
      : DLRModel(ctx, DLRBackend::kTVM) {
    SetupTVMModule(model_path);
  }

  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const char* GetInputName(int index) const override;
  virtual const char* GetInputType(int index) const override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, const void* input,
                        int dim) override;

  virtual void GetOutput(int index, void* out) override;
  virtual const void* GetOutputPtr(int index) const override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetOutputType(int index) const override;

  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  
  virtual void Run() override;
  virtual const char* GetBackend() const override;
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;

  /*
    Following methods use metadata file to lookup input and output names.
  */
  virtual const char* GetOutputName(const int index) const override;
  virtual int GetOutputIndex(const char* name) const override;
  virtual void GetOutputByName(const char* name, void* out) override;
};

}  // namespace dlr

#endif  // DLR_TVM_H_
