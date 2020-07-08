#ifndef DLR_TVM_H_
#define DLR_TVM_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/memory.h>

#include "dlr_model.h"

namespace dlr {

struct TVMModelArtifact : dlr::ModelArtifact {
  std::string model_lib;
  std::string params;
  std::string model_json;
};

class TVMModel : public DLRModel {
 private:
  tvm::runtime::ObjectPtr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::vector<const DLTensor*> outputs_;
  std::vector<std::string> output_types_;
  virtual void InitModelArtifact() override;
  void SetupTvmGraphRuntime();
  void FetchInputNodesData();
  void FetchOutputNodesData();
  void UpdateInputShapes();
  void UpdateOutputShapes();

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TVMModel(std::vector<std::string> paths, const DLContext& ctx)
      : DLRModel(paths, ctx, DLRBackend::kTVM) {
    InitModelArtifact();
    SetupTvmGraphRuntime();
    FetchInputNodesData();
    FetchOutputNodesData();
    LoadMetadataFromModelArtifact();
  }

  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const std::string& GetInputName(int index) const override;
  virtual const std::string& GetInputType(int index) const override;
  virtual const std::vector<int64_t>& GetInputShape(int index) const override;
  virtual void GetInput(int index, void* input) override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, void* input,
                        int dim) override;
  virtual void SetInput(const int index, const int64_t batch_size,
                        void* input) override;
  virtual void SetInput(std::string name, const int64_t batch_size,
                        void* input) override;

  virtual const int GetOutputDim(int index) const override;
  virtual const int64_t GetOutputSize(int index) const override;
  virtual const std::string& GetOutputName(const int index) const override;
  virtual const std::string& GetOutputType(int index) const override;
  virtual const std::vector<int64_t>& GetOutputShape(int index) const override;
  virtual int GetOutputIndex(const char* name) const override;
  virtual void GetOutput(int index, void* out) override;
  virtual void GetOutputByName(const char* name, void* out) override;

  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;

  virtual void Run() override;
};

}  // namespace dlr

#endif  // DLR_TVM_H_
