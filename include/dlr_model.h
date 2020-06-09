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

 public:
  DLRModel(const DLContext &ctx, const DLRBackend &backend)
      : ctx_(ctx), backend_(backend) {}
  virtual ~DLRModel() {}

  /* Input related functions */
  virtual int GetNumInputs() const { return num_inputs_; }
  virtual const char *GetInputName(int index) const = 0;
  virtual const char *GetInputType(int index) const = 0;
  virtual void GetInput(const char *name, void *input) = 0;
  virtual void SetInput(const char *name, const int64_t *shape, void *input,
                        int dim) = 0;

  /* Ouput related functions */
  virtual int GetNumOutputs() { return num_outputs_; }
  virtual const char *GetOutputName(const int index) const {
    LOG(ERROR) << "GetOutputName is not supported yet!";
  }
  virtual int GetOutputIndex(const char *name) const {
    LOG(ERROR) << "GetOutputName is not supported yet!";
  }
  virtual const char *GetOutputType(int index) const = 0;
  virtual void GetOutputShape(int index, int64_t *shape) const = 0;
  virtual void GetOutputSizeDim(int index, int64_t *size, int *dim) = 0;
  virtual void GetOutput(int index, void *out) = 0;
  virtual void GetOutputByName(const char *name, void *out) {
    LOG(ERROR) << "GetOutputByName is not supported yet!";
  }

  /* Weights releated functions */
  virtual int GetNumWeights() const { return num_weights_; }
  virtual const char *GetWeightName(int index) const = 0;
  virtual std::vector<std::string> GetWeightNames() const = 0;

  virtual const char *GetBackend() const = 0;
  virtual void SetNumThreads(int threads) = 0;
  virtual bool HasMetadata() const { return false; }
  virtual void UseCPUAffinity(bool use) = 0;
  virtual void Run() = 0;
};
}  // namespace dlr

#endif  // DLR_MODEL_H_
