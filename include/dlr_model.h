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
  size_t num_outputs_ = 1;
  nlohmann::json metadata;
  DLContext ctx_;
  std::vector<std::string> paths_;
  std::vector<std::string> input_names_;
  std::vector<std::string> input_types_;
  std::vector<std::string> output_names_;
  std::vector<std::string> output_types_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::shared_ptr<ModelArtifact> model_artifact_;

  virtual void InitModelArtifact() = 0;
  void LoadMetadataFromModelArtifact();
  void FetchOutputNamesFromMetadata();

 public:
  DLRModel(std::vector<std::string> paths, const DLContext &ctx,
           const DLRBackend &backend)
      : paths_(paths), ctx_(ctx), backend_(backend) {}
  virtual ~DLRModel() {}

  static DLRModel *create_model(std::string path, int device_type, int device_id);
  static DLRModel *create_model(std::vector<std::string> paths, int device_type, int device_id);
  static DLRModel *create_model(std::vector<std::string> paths, const DLContext &ctx);

  virtual int GetNumInputs() const { return num_inputs_; }
  virtual int GetNumOutputs() { return num_outputs_; }

  virtual const std::string GetBackend() const;

  virtual bool HasMetadata() const;

  virtual const int GetInputDim(int index) const = 0;
  virtual const int64_t GetInputSize(int index) const = 0;
  virtual const std::string &GetInputName(int index) const;
  virtual const std::vector<std::string> &GetInputNames() const;
  virtual const std::string &GetInputType(int index) const;
  virtual const std::vector<std::string> &GetInputTypes() const;
  virtual const std::vector<int64_t> &GetInputShape(int index) const;
  virtual void GetInput(int index, void *input) = 0;
  virtual void GetInput(const char *name, void *input) = 0;
  virtual void SetInput(const char *name, const int64_t *shape, void *input,
                        int dim) = 0;
  virtual void SetInput(const int index, int64_t batch_size, void *input) = 0;
  virtual void SetInput(std::string name, const int64_t batch_size,
                        void *input) = 0;

  virtual const int GetOutputDim(int index) const = 0;
  virtual const int64_t GetOutputSize(int index) const = 0;
  virtual const std::string &GetOutputName(const int index) const;
  virtual const std::vector<std::string> &GetOutputNames() const;
  virtual const std::string &GetOutputType(int index) const;
  virtual const std::vector<std::string> &GetOutputTypes() const;
  virtual const std::vector<int64_t> &GetOutputShape(int index) const;
  virtual int GetOutputIndex(const char *name) const;
  virtual void GetOutput(int index, void *out) = 0;
  virtual void GetOutputByName(const char *name, void *out);

  virtual void SetNumThreads(int threads) = 0;
  virtual void UseCPUAffinity(bool use) = 0;

  virtual void Run();
  virtual void Run(const int64_t batch_size, void **inputs, void **outputs);
  virtual void Run(const int64_t batch_size, std::map<std::string, void*>& inputs, std::vector<void*>& outputs);
};
}  // namespace dlr

#endif  // DLR_MODEL_H_
