#ifndef DLR_RELAYVM_H_
#define DLR_RELAYVM_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>
#include "dlr_common.h"


#ifdef _WIN32
#define LIBEXT ".dll"
#define LIBDLR "dlr.dll"
#elif __APPLE__
#define LIBEXT ".dylib"
#define LIBDLR "libdlr.dylib"
#else
#define LIBEXT ".so"
#define LIBDLR "libdlr.so"
#endif

#if defined(_MSC_VER) || defined(_WIN32)
#define DLR_DLL __declspec(dllexport)
#else
#define DLR_DLL
#endif  // defined(_MSC_VER) || defined(_WIN32)


namespace dlr {

class DLR_DLL RelayVMModel : public DLRModel {
 private:
  static const std::string ENTRY_FUNCTION;
  std::vector<std::string> output_names_;
  std::vector<std::string> output_types_;
  std::unique_ptr<ModelPath> path_;
  std::shared_ptr<tvm::runtime::Module> vm_module_;
  std::shared_ptr<tvm::runtime::Module> vm_executable_;
  std::vector<tvm::runtime::NDArray> inputs_;
  tvm::runtime::ObjectRef output_ref_;
  std::vector<tvm::runtime::NDArray> outputs_;
  std::vector<std::vector<int64_t>> output_shapes_;
  void InitModelPath(std::vector<std::string> paths);
  void SetupVMModule();
  void FetchInputNodesData();
  void LoadMetadata();
  void FetchOutputNodesData();
  void UpdateOutputs();
  void UpdateInputs();
  DLDataType GetInputDLDataType(int index);

 public:
  explicit RelayVMModel(std::vector<std::string> paths, const DLContext& ctx)
      : DLRModel(ctx, DLRBackend::kRELAYVM) {
    InitModelPath(paths);
    LoadMetadata();
    SetupVMModule();
    FetchInputNodesData();
    FetchOutputNodesData();
  }

  int GetInputIndex(const char* name) const;
  virtual const int GetInputDim(int index) const override;
  virtual const int64_t GetInputSize(int index) const override;
  virtual const char* GetInputName(int index) const override;
  virtual const char* GetInputType(int index) const override;
  virtual const char* GetWeightName(int index) const override;
  virtual std::vector<std::string> GetWeightNames() const override;
  virtual void GetInput(const char* name, void* input) override;
  virtual void SetInput(const char* name, const int64_t* shape, void* input, int dim) override;
  virtual void Run() override;
  tvm::runtime::NDArray GetOutput(int index);
  virtual void GetOutput(int index, void* out) override;
  virtual void GetOutputShape(int index, int64_t* shape) const override;
  virtual void GetOutputSizeDim(int index, int64_t* size, int* dim) override;
  virtual const char* GetOutputType(int index) const override;
  virtual const char* GetBackend() const override;
  virtual void SetNumThreads(int threads) override;
  virtual void UseCPUAffinity(bool use) override;

  /*
    Following methods use metadata file to lookup input and output names.
  */
  virtual const char* GetOutputName(const int index) const override;
  virtual int GetOutputIndex(const char* name) const override;
  virtual void GetOutputByName(const char* name, void* out) override;
};

}  // namespace dlr

#endif  // DLR_RELAYVM_H_
