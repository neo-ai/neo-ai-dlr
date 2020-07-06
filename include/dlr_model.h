#ifndef DLR_MODEL_H_
#define DLR_MODEL_H_

#include "dlr_common.h"

namespace dlr {
// Abstract class
class DLRModel {
 protected:
  std::string version_;
  DLRBackend backend_;
  size_t num_inputs_ = 1;
  size_t num_weights_ = 0;
  size_t num_outputs_ = 1;
  DLContext ctx_;
  std::vector<std::string> input_names_;
  std::vector<std::string> input_types_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;

 public:
  DLRModel(const DLContext &ctx, const DLRBackend &backend)
      : ctx_(ctx), backend_(backend) {}
  virtual ~DLRModel() {}

  /* Input related functions */
  virtual int GetNumInputs() const { return num_inputs_; }
  virtual const std::string& GetInputName(int index) const = 0;
  virtual const std::string& GetInputType(int index) const = 0;

  virtual const std::vector<int64_t>& GetInputShape(int index) const = 0;
  virtual const int64_t GetInputSize(int index) const = 0;
  virtual const int GetInputDim(int index) const = 0;

  virtual void SetInput(std::string name, const int batch_size, void* input) = 0;
  virtual void SetInput(const char *name, const int64_t *shape, void *input,
                        int dim) = 0;
  virtual void GetInput(const char *name, void *input) = 0;

  /* Ouput related functions */
  virtual int GetNumOutputs() { return num_outputs_; }
  virtual const std::string& GetOutputName(const int index) const {
    LOG(ERROR) << "GetOutputName is not supported yet!";
  }
  virtual int GetOutputIndex(const char *name) const {
    LOG(ERROR) << "GetOutputName is not supported yet!";
  }
  virtual const std::string& GetOutputType(int index) const = 0;
  virtual const std::vector<int64_t>& GetOutputShape(int index) const = 0;
  virtual const int64_t GetOutputSize(int index) const = 0;
  virtual const int GetOutputDim(int index) const = 0;
  virtual void GetOutput(int index, void *out) = 0;
  virtual void GetOutputByName(const char *name, void *out) {
    LOG(ERROR) << "GetOutputByName is not supported yet!";
  }

  /* Weights releated functions */
  virtual int GetNumWeights() const { return num_weights_; }
  virtual const std::string& GetWeightName(int index) const = 0;

  virtual void SetNumThreads(int threads) = 0;
  virtual bool HasMetadata() const { return false; }
  virtual void UseCPUAffinity(bool use) = 0;
  virtual void Run() = 0;

  virtual const std::string GetBackend() const {
    if (backend_ == DLRBackend::kTVM) {
      return "tvm";
    } else if (backend_ == DLRBackend::kTREELITE) {
      return "treelite";
    } else if (backend_ == DLRBackend::kTREELITE) {
      return "hexagon";
    } else {
      LOG(ERROR) << "Unsupported DLRBackend!";
    }
  };
};
}  // namespace dlr

#endif  // DLR_MODEL_H_
