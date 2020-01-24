#ifndef DLR_TENSORFLOW_H_
#define DLR_TENSORFLOW_H_

#include "dlr_common.h"
#include "tensorflow/c/c_api.h"

namespace dlr {

/*! \brief Get the paths of the Tensorflow model files.
 */
std::string GetTensorflowFile(const std::string& dirname);

/*! \brief free_buffer function used to cleanup memory after TF model is built.
 */
void FreeBuffer(void* data, size_t length);

/*! \brief read tensorflow model file.
 */
TF_Buffer* ReadTFFile(const char* file);

/*! \brief class TensorflowModel
 */
class TensorflowModel : public DLRModel {
 private:
  TF_Status* status_;
  TF_Graph* graph_;
  TF_Session* sess_;
  // input_names_ are declared in base class
  std::vector<std::string> output_names_;
  std::vector<TF_Output> inputs_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> input_tensors_;
  std::vector<TF_Tensor*> output_tensors_;
  void LoadFrozenModel(const char* pb_file);
  void GenTensorSpec(bool is_input, const int batch_size);
  int GetInputId(const char* name);

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TensorflowModel(const std::string& model_path, const DLContext& ctx,
                           const std::vector<std::string>& inputs,
                           const std::vector<std::string>& outputs,
                           const int batch_size, const int threads);
  ~TensorflowModel();

  virtual const char* GetInputName(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, float* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, float* input,
                        int dim) override;
  virtual void Run() override;
  virtual void GetOutput(int index, float* out) override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetBackend() const override;
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
};

}  // namespace dlr

#endif  // DLR_TENSORFLOW_H_
