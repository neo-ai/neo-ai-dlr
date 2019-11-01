#ifndef DLR_TREELITE_H_
#define DLR_TREELITE_H_

#include <treelite/c_api_runtime.h>
#include "dlr_common.h"


namespace dlr {

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

/*! \brief Get the paths of the Treelite model files.
 */
ModelPath GetTreelitePaths(const std::string& dirname);

/*! \brief class TreeliteModel
 */
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
  /*! \brief Load model files from given folder path.
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
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
  virtual void GetRuntimeEnabled(const char* device, bool* enabled) override;
};

} // namespace dlr


#endif  // DLR_TREELITE_H_
