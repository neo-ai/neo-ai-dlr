#ifndef DLR_HEXAGON_H_
#define DLR_HEXAGON_H_

#include "dlr_model.h"

namespace dlr {

struct HexagonModelArtifact: ModelArtifact {
  std::string model_file;
  std::string skeleton_file;
};

typedef struct {
  std::string name;
  int dim;
  std::vector<int> shape;
  int64_t size;
  size_t bytes;
} HexagonTensorSpec;

/*! \brief class HexagonModel
 */
class HexagonModel : public DLRModel {
 private:
  std::vector<HexagonTensorSpec> input_tensors_spec_;
  std::vector<HexagonTensorSpec> output_tensors_spec_;
  int graph_id_ = 0;
  int debug_level_ = 0;
  uint8_t* input_ = nullptr;
  uint8_t* output_ = nullptr;
  char* log_buf_ = nullptr;

  int (*dlr_hexagon_model_init)(int*, uint8_t**, uint8_t**, int);
  int (*dlr_hexagon_model_exec)(int, uint8_t*, uint8_t*);
  void (*dlr_hexagon_model_close)(int);
  int (*dlr_hexagon_nn_getlog)(int, unsigned char*, int);
  int (*dlr_hexagon_input_spec)(int, char**, int*, int**, int*, int*);
  int (*dlr_hexagon_output_spec)(int, char**, int*, int**, int*, int*);

  virtual void InitModelArtifact() override;
  void LoadSymbols();
  void InitHexagonModel();
  void PrintHexagonNNLog();
  void AllocateLogBuffer();
  void GenTensorSpec(bool isInput);
  void InitInputOutputTensorSpecs();
  int GetInputId(const char* name);
  void UpdateInputShapes();
  void UpdateOutputShapes();

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit HexagonModel(std::vector<std::string> paths, const DLContext& ctx, const int debug_level)
      : DLRModel(paths, ctx, DLRBackend::kHEXAGON), debug_level_(debug_level) {
    InitModelArtifact();
    AllocateLogBuffer();
    LoadSymbols();
    InitHexagonModel();
    InitInputOutputTensorSpecs();
    LoadMetadataFromModelArtifact();
  }

  ~HexagonModel();

  static const int kLogBufferSize = 2*1024*1024;

  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const std::string& GetInputName(int index) const override;
  virtual const std::string& GetInputType(int index) const override;
  virtual const std::vector<int64_t>& GetInputShape(int index) const override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, void* input, int dim) override;
  virtual void SetInput(const int index, const int64_t batch_size, void* input) override;
  virtual void SetInput(std::string name, const int64_t batch_size, void* input) override;

  virtual const int GetOutputDim(int index) const override;
  virtual const int64_t GetOutputSize(int index) const override;
  virtual const std::string& GetOutputType(int index) const override;
  virtual const std::vector<int64_t>& GetOutputShape(int index) const override;
  virtual void GetOutput(int index, void* out) override;

  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;

  virtual void Run() override;
  // virtual void Run(const int batch_size, std::vector<void*> intputs, std::vector<void*> outputs) override;
};

}  // namespace dlr

#endif  // DLR_HEXAGON_H_
