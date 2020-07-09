#ifndef DLR_TREELITE_H_
#define DLR_TREELITE_H_

#include <treelite/c_api_runtime.h>

#include "dlr_model.h"

namespace dlr {

struct TreeliteModelArtifact : ModelArtifact {
  std::string model_lib;
};

/*! \brief Structure to hold Treelite Input.
 */
struct TreeliteInput {
  std::vector<float> data;
  std::vector<uint32_t> col_ind;
  std::vector<size_t> row_ptr;
  size_t num_row;
  size_t num_col;
  CSRBatchHandle handle;
};

/*! \brief class TreeliteModel
 */
class TreeliteModel : public DLRModel {
 private:
  static const std::string INPUT_NAME;
  static const std::string INPUT_TYPE;
  static const std::string OUTPUT_TYPE;
  // fields for Treelite model
  PredictorHandle model_;
  size_t num_of_input_features_;
  // size of temporary buffer per instance
  size_t output_buffer_size_;
  // size of output per instance
  size_t output_size_;
  std::unique_ptr<TreeliteInput> input_ = nullptr;
  std::vector<float> output_ = {};
  virtual void InitModelArtifact() override;
  void SetupTreeliteModel();
  void FetchModelNodesData();
  void UpdateInputShapes();
  void UpdateOutputShapes();
  static const int kInputDim = 2;

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TreeliteModel(std::vector<std::string> paths, const DLContext& ctx)
      : DLRModel(paths, ctx, DLRBackend::kTREELITE) {
    InitModelArtifact();
    SetupTreeliteModel();
    FetchModelNodesData();
    LoadMetadataFromModelArtifact();
  }

  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
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
  virtual void GetOutput(int index, void* out) override;

  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;

  virtual void Run() override;
};

}  // namespace dlr

#endif  // DLR_TREELITE_H_
