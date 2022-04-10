#ifndef DLR_TENSORFLOW2_H_
#define DLR_TENSORFLOW2_H_

#include "dlr.h"
#include "dlr_common.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace dlr {

/*! \brief convert DLR_TF2Config to protobuf vector of bytes
 */
void PrepareTF2ConfigProto(const DLR_TF2Config& tf2_config, std::vector<std::uint8_t>& config);

/*! \brief class Tensorflow2Model
 */
class Tensorflow2Model : public DLRModel {
  typedef google::protobuf::Map<std::string, tensorflow::TensorInfo> InputOutputType;

 private:
  TF_Status* status_;
  TF_Graph* graph_;
  TF_Session* sess_;
  std::vector<std::vector<int64_t>> graph_input_shapes_;  // might have -1 dimensions
  std::vector<std::string> output_names_;
  std::vector<std::string> output_types_;
  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
  std::vector<TF_Output> inputs_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> input_tensors_;
  std::vector<TF_Tensor*> output_tensors_;
  TF_Output ParseTensorName(const std::string& t_name);
  void DetectInputsAndOutputs(const InputOutputType& inputs, const InputOutputType& outputs);
  void PrepInputs();
  void PrepOutputs();
  int GetInputId(const char* name);
  TF_Tensor* AllocateInputTensor(int index, const int64_t* dims, const int n_dim);

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit Tensorflow2Model(const std::string& model_path, const DLDevice& dev,
                            const DLR_TF2Config& tf2_config);
  ~Tensorflow2Model();

  virtual const char* GetInputName(int index) const override;
  virtual const char* GetInputType(int index) const override;
  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const std::vector<int64_t>& GetInputShape(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, const void* input,
                        int dim) override;
  virtual void Run() override;
  virtual void GetOutput(int index, void* out) override;
  virtual int GetOutputIndex(const char* name) const override;
  virtual void GetOutputByName(const char* name, void* out) override;
  virtual const void* GetOutputPtr(int index) const override;
  virtual const char* GetOutputName(const int index) const override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetOutputType(int index) const override;
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
};

}  // namespace dlr

#endif  // DLR_TENSORFLOW2_H_
