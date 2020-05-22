#ifndef DLR_TFLITE_H_
#define DLR_TFLITE_H_

#include "dlr_common.h"
#include "tensorflow/lite/kernels/register.h"

namespace dlr {

/*! \brief Get the paths of the TFLite model files.
 */
std::string GetTFLiteFile(const std::string& dirname);

typedef struct {
  int id;
  std::string name;
  TfLiteType type;
  int dim;
  std::vector<int> shape;
  int64_t size;
  size_t bytes;
} TensorSpec;

/*! \brief class TFLiteModel
 */
class TFLiteModel : public DLRModel {
 private:
  tflite::StderrReporter* error_reporter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::vector<TensorSpec> input_tensors_spec_;
  std::vector<TensorSpec> output_tensors_spec_;
  void GenTensorSpec(bool isInput);
  int GetInputId(const char* name);

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit TFLiteModel(const std::string& model_path, const DLContext& ctx,
                       const int threads, const bool use_nnapi);
  ~TFLiteModel();

  virtual const char* GetInputName(int index) const override;
  virtual const char* GetInputType(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, void* input,
                        int dim) override;
  virtual void Run() override;
  virtual void GetOutput(int index, void* out) override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetOutputType(int index) const override;
  virtual const char* GetBackend() const override;
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
};

}  // namespace dlr

#endif  // DLR_TFLITE_H_
