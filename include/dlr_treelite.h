#ifndef DLR_TREELITE_H_
#define DLR_TREELITE_H_

#include <treelite/c_api_runtime.h>

#include "dlr_allocator.h"
#include "dlr_common.h"

#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL __declspec(dllexport)
#else
#define DLR_DLL
#endif  // defined(_MSC_VER) || defined(_WIN32)

namespace dlr {

/*! \brief Structure to hold Treelite Input.
 */
struct TreeliteInput {
  std::vector<float, DLRAllocator<float>> data;
  std::vector<uint32_t, DLRAllocator<uint32_t>> col_ind;
  std::vector<size_t, DLRAllocator<size_t>> row_ptr;
  size_t num_row;
  size_t num_col;
  DMatrixHandle handle;
};

/*! \brief Get the paths of the Treelite model files.
 */
ModelPath SetTreelitePaths(const std::vector<std::string>& files);

/*! \brief class TreeliteModel
 */
class DLR_DLL TreeliteModel : public DLRModel {
 private:
  static const std::string INPUT_NAME;
  static const std::string INPUT_TYPE;
  static const std::string OUTPUT_TYPE;
  static const int kInputDim = 2;
  // fields for Treelite model
  PredictorHandle treelite_model_;
  size_t treelite_num_feature_;
  // size of temporary buffer per instance
  size_t treelite_output_buffer_size_;
  // size of output per instance
  size_t treelite_output_size_;
  std::unique_ptr<TreeliteInput> treelite_input_;
  std::vector<float, DLRAllocator<float>> treelite_output_;
  /*! \brief Whether input is sparse (zero values should be skipped) */
  bool has_sparse_input_;
  void SetupTreeliteModule(const std::vector<std::string>& files);
  void UpdateInputShapes();

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TreeliteModel(const std::vector<std::string>& files, const DLContext& ctx)
      : DLRModel(ctx, DLRBackend::kTREELITE) {
    SetupTreeliteModule(files);
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
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
};

}  // namespace dlr

#endif  // DLR_TREELITE_H_
