#ifndef DLR_PIPELINE_H_
#define DLR_PIPELINE_H_

#include <graph/graph_runtime.h>
#include <tvm/runtime/memory.h>
#include "dlr_common.h"

#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL __declspec(dllexport)
#else
#define DLR_DLL
#endif  // defined(_MSC_VER) || defined(_WIN32)

namespace dlr {

/*! \brief class PipelineModel
 */
class DLR_DLL PipelineModel : public DLRModel {
 private:
  int count_;
  const std::vector<DLRModelPtr> dlr_models_;
  void CheckModelsCompatibility(const DLRModelPtr& m0, const DLRModelPtr& m1, const int m1_id,
                                const bool is_runtime_check);
  void SetupPipelineModel();

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit PipelineModel(const std::vector<DLRModelPtr>& dlr_models, const DLContext& ctx)
      : DLRModel(ctx, DLRBackend::kPIPELINE), dlr_models_(dlr_models) {
    SetupPipelineModel();
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

#endif  // DLR_PIPELINE_H_
