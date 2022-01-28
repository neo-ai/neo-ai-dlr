#ifndef DLR_HEXAGON_H_
#define DLR_HEXAGON_H_

#include "dlr_common.h"

namespace dlr {

/*! \brief Get the paths of the Hexagon model files.
 */
std::string GetHexagonModelFile(const std::vector<std::string>& files);
bool FindHexagonNNSkelFile(const std::string& dirname);
void* FindSymbol(void* handle, const char* fn_name);

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
  void GenTensorSpec(bool isInput);
  int GetInputId(const char* name);
  void PrintHexagonNNLog();
  void UpdateInputShapes();
  int graph_id_;
  uint8_t* input_;
  uint8_t* output_;
  int debug_level_;
  char* log_buf_;
  int log_buf_size_;

  int (*dlr_hexagon_model_init)(int*, uint8_t**, uint8_t**, int);
  int (*dlr_hexagon_model_exec)(int, uint8_t*, uint8_t*);
  void (*dlr_hexagon_model_close)(int);
  int (*dlr_hexagon_nn_getlog)(int, unsigned char*, int);
  int (*dlr_hexagon_input_spec)(int, char**, int*, int**, int*, int*);
  int (*dlr_hexagon_output_spec)(int, char**, int*, int**, int*, int*);

 public:
  /*! \brief Load model files from given folder path.
   */
  explicit HexagonModel(const std::vector<std::string>& files, const DLDevice& dev,
                        const int debug_level);
  ~HexagonModel();

  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const char* GetInputName(int index) const override;
  virtual const char* GetInputType(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, const void* input,
                        int dim) override;
  virtual void Run() override;
  virtual void GetOutput(int index, void* out) override;
  virtual const void* GetOutputPtr(int index) const override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetOutputType(int index) const override;
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;
};

}  // namespace dlr

#endif  // DLR_HEXAGON_H_
