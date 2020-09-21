#include "dlr_relayvm.h"
#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <numeric>

using namespace dlr;

const std::string RelayVMModel::ENTRY_FUNCTION = "main";

void RelayVMModel::InitModelPath(std::vector<std::string> paths) {
  path_ = std::make_unique<ModelPath>();
  std::vector<std::string> paths_vec;
  for (auto path : paths) {
    ListDir(path, paths_vec);
  }

  for (auto path : paths_vec) {
    if (!EndsWith(path, LIBDLR) && EndsWith(path, ".so")) {
      path_->model_lib = path;
    } else if (EndsWith(path, ".ro")) {
      path_->relay_executable = path;
    } else if (EndsWith(path, ".meta")) {
      path_->metadata = path;
    }
  }

  if (path_->model_lib.empty() || path_->relay_executable.empty() || path_->metadata.empty()) {
    LOG(FATAL) << "Invalid model artifact!";
  }
}

void RelayVMModel::SetupVMModule() {
  tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(path_->model_lib);
  std::ifstream relay_ob(path_->relay_executable, std::ios::binary);
  std::string code_data((std::istreambuf_iterator<char>(relay_ob)),
                        std::istreambuf_iterator<char>());

  vm_executable_ =
      std::make_shared<tvm::runtime::Module>(tvm::runtime::vm::Executable::Load(code_data, lib));
  auto vm = tvm::runtime::make_object<tvm::runtime::vm::VirtualMachine>();
  vm->LoadExecutable(static_cast<tvm::runtime::vm::Executable*>(
      const_cast<tvm::runtime::Object*>(vm_executable_->get())));
  vm_module_ = std::make_shared<tvm::runtime::Module>(tvm::runtime::Module(vm));

  tvm::runtime::PackedFunc init = vm_module_->GetFunction("init");
  init(static_cast<int>(ctx_.device_type), ctx_.device_id, static_cast<int>(tvm::runtime::vm::AllocatorType::kPooled));
}

void RelayVMModel::LoadMetadata() { 
  LoadJsonFromFile(path_->metadata, metadata_); 
  ValidateDeviceTypeIfExists();
}

void RelayVMModel::FetchInputNodesData() {
  tvm::runtime::vm::Executable* exec = static_cast<tvm::runtime::vm::Executable*>(
      const_cast<tvm::runtime::Object*>(vm_executable_->get()));
  num_inputs_ = exec->GetFunctionArity(ENTRY_FUNCTION);
  input_names_.resize(num_inputs_);
  input_types_.resize(num_inputs_);
  input_shapes_.resize(num_inputs_);
  inputs_.resize(num_inputs_);
  for (int i = 0; i < num_inputs_; i++) {
    input_names_[i] = exec->GetFunctionParameterName(ENTRY_FUNCTION, i);
    for (auto shape : metadata_.at("Model").at("Inputs").at(i).at("shape")) {
      if (shape == nullptr) {
        input_shapes_[i].push_back(-1);
      } else {
        input_shapes_[i].push_back(shape);
      }
    };
  }
  try {
    for (int i = 0; i < num_inputs_; i++) {
      input_types_[i] = metadata_.at("Model").at("Inputs").at(i).at("dtype");
    }
  } catch (nlohmann::json::out_of_range& e) {
    LOG(ERROR) << e.what();
    throw dmlc::Error("No Input types metadata found!");
  }
}

void RelayVMModel::FetchOutputNodesData() {
  try {
    num_outputs_ = metadata_.at("Model").at("Outputs").size();
  } catch (nlohmann::json::out_of_range& e) {
    LOG(ERROR) << e.what();
    throw dmlc::Error("No Output metadata found!");
  }
  output_names_.resize(num_outputs_);
  output_types_.resize(num_outputs_);
  output_shapes_.resize(num_outputs_);
  try {
    for (int i = 0; i < num_outputs_; i++) {
      output_names_[i] = metadata_.at("Model").at("Outputs").at(i).at("name");
      output_types_[i] = metadata_.at("Model").at("Outputs").at(i).at("dtype");
      for (auto shape : metadata_.at("Model").at("Outputs").at(i).at("shape")) {
        if (shape == nullptr) {
          output_shapes_[i].push_back(-1);
        } else {
          output_shapes_[i].push_back(shape);
        }
      };
    }
  } catch (nlohmann::json::out_of_range& e) {
    LOG(ERROR) << e.what();
    throw dmlc::Error("No Output metadata found!");
  }
}

const char* RelayVMModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* RelayVMModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_types_[index].c_str();
}

const char* RelayVMModel::GetWeightName(int index) const { throw dmlc::Error("Not Implemented!"); }

std::vector<std::string> RelayVMModel::GetWeightNames() const {
  throw dmlc::Error("Not Implemented!");
}

void RelayVMModel::GetInput(const char* name, void* input) {
  int index = GetInputIndex(name);
  auto in_array = inputs_[index];
  DLTensor input_tensor;
  input_tensor.data = input;
  input_tensor.ctx = ctx_;
  input_tensor.ndim = in_array->ndim;
  input_tensor.dtype = in_array->dtype;
  input_tensor.shape = in_array->shape;
  input_tensor.strides = nullptr;
  input_tensor.byte_offset = 0;
  in_array.CopyTo(&input_tensor);
}

int RelayVMModel::GetInputIndex(const char* name) const {
  std::string input_name(name);
  for (auto i = 0; i < num_inputs_; i++) {
    if (input_name == input_names_[i]) {
      return i;
    }
  }
  throw dmlc::Error("Invalid input node name!");
}

const int RelayVMModel::GetInputDim(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index].size();
}

const int64_t RelayVMModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return std::accumulate(input_shapes_[index].begin(), input_shapes_[index].end(), 1,
                         std::multiplies<int64_t>());
}

DLDataType RelayVMModel::GetInputDLDataType(int index) {
  auto input_type = input_types_[index];
  DLDataType dtype;
  dtype.lanes = 1;
  if (input_type == "uint8") {
    dtype.code = kDLUInt;
    dtype.bits = 8;
  } else if (input_type == "int8") {
    dtype.code = kDLInt;
    dtype.bits = 8;
  } else if (input_type == "float32") {
    dtype.code = kDLFloat;
    dtype.bits = 32;
  } else if (input_type == "float64") {
    dtype.code = kDLBfloat;
    dtype.bits = 64;
  } else {
    throw dmlc::Error("Unknown input dtype!");
  }
  return dtype;
}

void RelayVMModel::SetInput(const char* name, const int64_t* shape, void* input, int dim) {
  int index = GetInputIndex(name);
  DLDataType dtype = GetInputDLDataType(index);
  DLTensor input_tensor;
  input_tensor.data = input;
  input_tensor.ctx = ctx_;
  input_tensor.ndim = dim;
  input_tensor.shape = const_cast<int64_t*>(shape);
  input_tensor.strides = nullptr;
  input_tensor.byte_offset = 0;
  input_tensor.dtype = dtype;
  std::vector<int64_t> arr_shape(shape, shape + dim);

  tvm::runtime::NDArray input_arr = tvm::runtime::NDArray::Empty(arr_shape, dtype, ctx_);
  input_arr.CopyFrom(&input_tensor);
  inputs_[index] = input_arr;
}

void RelayVMModel::UpdateInputs() {
  const int kNumArgs = GetNumInputs() + 1;
  TVMValue *values = (TVMValue*)malloc(sizeof(TVMValue) * kNumArgs);
  int *type_codes = (int*)malloc(sizeof(int) * kNumArgs);
  auto arg_setter = tvm::runtime::TVMArgsSetter(values, type_codes);
  arg_setter(0, ENTRY_FUNCTION);
  for (int i = 0; i < inputs_.size(); i++) {
    arg_setter(i + 1, inputs_[i]);
  }

  tvm::runtime::PackedFunc set_input = vm_module_->GetFunction("set_input");
  tvm::runtime::TVMRetValue rv;
  set_input.CallPacked(tvm::runtime::TVMArgs(values, type_codes, kNumArgs), &rv);

  free(values);
  free(type_codes);
}

void RelayVMModel::Run() {
  // Invoke inference
  UpdateInputs();
  tvm::runtime::PackedFunc invoke = vm_module_->GetFunction("invoke");
  output_ref_ = invoke(ENTRY_FUNCTION);
  UpdateOutputs();
}

void RelayVMModel::UpdateOutputs() {
  outputs_.resize(num_outputs_);
  if (output_ref_->IsInstance<tvm::runtime::ADTObj>()) {
    auto adt = tvm::runtime::Downcast<tvm::runtime::ADT>(output_ref_);
    for (int i = 0; i < adt.size(); i++) {
      outputs_[i] = tvm::runtime::Downcast<tvm::runtime::NDArray>(adt[i]);
    }
  } else if (output_ref_->IsInstance<tvm::runtime::NDArray::ContainerType>()) {
    outputs_[0] = tvm::runtime::Downcast<tvm::runtime::NDArray>(output_ref_);
  } else {
    throw dmlc::Error("Invalid output_ref format!");
  }
}

void RelayVMModel::GetOutput(int index, void* output) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  auto out_array = outputs_[index];
  DLTensor output_tensor;
  output_tensor.data = output;
  output_tensor.ctx = ctx_;
  output_tensor.ndim = out_array->ndim;
  output_tensor.dtype = out_array->dtype;
  output_tensor.shape = out_array->shape;
  output_tensor.strides = nullptr;
  output_tensor.byte_offset = 0;
  out_array.CopyTo(&output_tensor);
}

void RelayVMModel::GetOutputShape(int index, int64_t* shape) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  if (outputs_.empty()) {
    // Inference has not been called yet. Get shapes from metadata.
    CHECK_LT(index, output_shapes_.size()) << "Output index is out of range.";
    std::copy(output_shapes_[index].begin(), output_shapes_[index].end(), shape);
  } else {
    CHECK_LT(index, outputs_.size()) << "Output index is out of range.";
    std::memcpy(shape, outputs_[index]->shape, sizeof(int64_t) * outputs_[index]->ndim);
  }
}

void RelayVMModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  CHECK_LT(index, output_shapes_.size()) << "Output index is out of range.";
  *size = 1;
  if (index < outputs_.size()) {
    auto arr = outputs_[index];
    *size = std::accumulate(arr->shape, arr->shape + arr->ndim, 1, std::multiplies<int64_t>());
    *dim = arr->ndim;
  } else {
    *size = std::accumulate(output_shapes_[index].begin(), output_shapes_[index].end(), 1, std::multiplies<int64_t>());
    *dim = output_shapes_[index].size();
  }
}

const char* RelayVMModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_types_[index].c_str();
}

const char* RelayVMModel::GetBackend() const { return "relayvm"; }

void RelayVMModel::SetNumThreads(int threads) { throw dmlc::Error("Not Implemented!"); }

void RelayVMModel::UseCPUAffinity(bool use) { throw dmlc::Error("Not Implemented!"); }

const char* RelayVMModel::GetOutputName(const int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_names_[index].c_str();
}

int RelayVMModel::GetOutputIndex(const char* name) const {
  if (!this->HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  for (int i = 0; i < this->num_outputs_; i++) {
    const char* output_name = GetOutputName(i);
    if (output_name == nullptr) return -1;
    if (strcmp(output_name, name) == 0) {
      return i;
    }
  }

  std::string msg = "Couldn't find index for output node";
  msg += " " + std::string{name} + "!";
  throw dmlc::Error(msg);
}

void RelayVMModel::GetOutputByName(const char* name, void* out) {
  int output_index = this->GetOutputIndex(name);
  this->GetOutput(output_index, out);
}
