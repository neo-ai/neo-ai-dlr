#ifndef DLR_TVM_H_
#define DLR_TVM_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/memory.h>
#include "dlr_model.h"

namespace dlr {

struct TVMModelArtifact: dlr::ModelArtifact {
  std::string model_lib;
  std::string params;
  std::string model_json;
};


class TVMModel : public DLRModel {
 private:
  tvm::runtime::ObjectPtr<tvm::runtime::GraphRuntime> tvm_graph_runtime_;
  std::shared_ptr<tvm::runtime::Module> tvm_module_;
  std::vector<const DLTensor*> outputs_;
  std::vector<std::string> output_types_;
  std::vector<std::string> weight_names_;
  TVMModelArtifact model_artifact_;
  nlohmann::json metadata;
  void InitModelArtifact(const std::vector<std::string> &paths);
  void SetupTvmGraphRuntimeAndModule();
  void LoadModelMetadata();
  void FetchModelNodesData();
  void FetchInputAndWeightNodesData();
  void FetchOutputNodesData();
  void UpdateInputShapes();
  void UpdateOutputShapes();

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TVMModel(std::vector<std::string> paths, const DLContext& ctx)
      : DLRModel(ctx, DLRBackend::kTVM) {
    InitModelArtifact(paths);
    SetupTvmGraphRuntimeAndModule();
    LoadModelMetadata();
    FetchModelNodesData();
  }

  virtual const std::string& GetInputName(int index) const override;
  virtual const std::string& GetInputType(int index) const override;

  virtual const std::vector<int64_t>& GetInputShape(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const int GetInputDim(int index) const override;

  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, void* input,
                        int dim) override;

  virtual void GetOutput(int index, void* out) override;
  virtual const std::vector<int64_t>& GetOutputShape(int index) const override;
  virtual const int64_t GetOutputSize(int index) const override;
  virtual const int GetOutputDim(int index) const override;
  virtual const std::string& GetOutputType(int index) const override;
  virtual const std::string& GetOutputName(const int index) const override;
  virtual int GetOutputIndex(const char* name) const override;
  virtual void GetOutputByName(const char* name, void* out) override;

  virtual const std::string& GetWeightName(int index) const override;

  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;

  virtual void Run() override;

  /*
    Following methods use metadata file to lookup input and output names.
  */
  virtual bool HasMetadata() const override;
};

}  // namespace dlr

#endif  // DLR_TVM_H_
