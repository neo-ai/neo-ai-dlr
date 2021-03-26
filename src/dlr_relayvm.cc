#include "dlr_relayvm.h"

#include <stdlib.h>

#include <fstream>
#include <iterator>
#include <numeric>

using namespace dlr;

const std::string RelayVMModel::ENTRY_FUNCTION = "main";

void RelayVMModel::SetupVMModule(const std::vector<std::string>& files) {
  ModelPath path;
  dlr::InitModelPath(files, &path);
  if (path.model_lib.empty() || path.relay_executable.empty() || path.metadata.empty()) {
    throw dmlc::Error("Invalid RelayVM model artifact. Must have .so, .ro, and .meta files.");
  }

  const std::vector<DLRModelElem> model_elems = {
      {DLRModelElemType::RELAY_EXEC, path.relay_executable.c_str(), nullptr, 0},
      {DLRModelElemType::TVM_LIB, path.model_lib.c_str(), nullptr, 0},
      {DLRModelElemType::NEO_METADATA, path.metadata.c_str(), nullptr, 0}};
  SetupVMModule(model_elems);
}

void RelayVMModel::SetupVMModule(const std::vector<DLRModelElem>& model_elems) {
  // Set custom allocators in TVM.
  if (dlr::DLRAllocatorFunctions::GetMemalignFunction() &&
      dlr::DLRAllocatorFunctions::GetFreeFunction()) {
    auto* pf = tvm::runtime::Registry::Get("runtime.contrib.set_custom_cpu_allocator");
    if (pf) {
      (*pf)(reinterpret_cast<void*>(dlr::DLRAllocatorFunctions::GetMemalignFunction()),
            reinterpret_cast<void*>(dlr::DLRAllocatorFunctions::GetFreeFunction()));
    } else {
      LOG(WARNING) << "Custom allocator functions are not available. Using default allocators.";
    }
  } else if (dlr::DLRAllocatorFunctions::AnySet()) {
    LOG(WARNING) << "SetDLRCustomAllocatorFree() and SetDLRCustomAllocatorMemalign() must be set "
                    "to override TVM allocations. Using default allocators.";
  }

  std::string code_data;
  std::string model_lib_path;
  std::string metadata_data;
  for (DLRModelElem el : model_elems) {
    if (el.type == DLRModelElemType::RELAY_EXEC) {
      if (el.path != nullptr) {
        code_data = dlr::LoadFileToString(el.path, std::ios::binary);
      } else if (el.data != nullptr && el.data_size > 0) {
        code_data.assign(static_cast<const char*>(el.data), el.data_size);
      } else {
        throw dmlc::Error("Invalid RelayVM model element RELAY_EXEC");
      }
    } else if (el.type == DLRModelElemType::TVM_LIB) {
      if (el.path != nullptr) {
        model_lib_path = el.path;
      } else {
        throw dmlc::Error("Invalid RelayVM model element TVM_LIB. TVM_LIB must be a file path.");
      }
    } else if (el.type == DLRModelElemType::NEO_METADATA) {
      if (el.path != nullptr) {
        metadata_data = dlr::LoadFileToString(el.path);
      } else if (el.data != nullptr) {
        metadata_data = static_cast<const char*>(el.data);
      } else {
        throw dmlc::Error("Invalid model element NEO_METADATA");
      }
    }
  }
  if (code_data.empty() || model_lib_path.empty() || metadata_data.empty()) {
    throw dmlc::Error(
        "Invalid RelayVM model. Must have RELAY_EXEC, TVM_LIB and NEO_METADATA elements");
  }

  LoadJsonFromString(metadata_data, this->metadata_);
  ValidateDeviceTypeIfExists();

  tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(model_lib_path);

  vm_executable_ =
      std::make_shared<tvm::runtime::Module>(tvm::runtime::vm::Executable::Load(code_data, lib));
  auto vm = tvm::runtime::make_object<tvm::runtime::vm::VirtualMachine>();
  vm->LoadExecutable(static_cast<tvm::runtime::vm::Executable*>(
      const_cast<tvm::runtime::Object*>(vm_executable_->get())));
  vm_module_ = std::make_shared<tvm::runtime::Module>(tvm::runtime::Module(vm));

  tvm::runtime::PackedFunc init = vm_module_->GetFunction("init");
  if (ctx_.device_type == DLDeviceType::kDLCPU) {
    init(static_cast<int>(ctx_.device_type), ctx_.device_id,
         static_cast<int>(tvm::runtime::vm::AllocatorType::kPooled));
  } else {
    // CPU context also must be initialized because input/output data comes from CPU.
    init(static_cast<int>(ctx_.device_type), ctx_.device_id,
         static_cast<int>(tvm::runtime::vm::AllocatorType::kPooled),
         static_cast<int>(DLDeviceType::kDLCPU), 0,
         static_cast<int>(tvm::runtime::vm::AllocatorType::kPooled));
  }
}

void RelayVMModel::FetchInputNodesData() {
  tvm::runtime::vm::Executable* exec = static_cast<tvm::runtime::vm::Executable*>(
      const_cast<tvm::runtime::Object*>(vm_executable_->get()));
  num_inputs_ = exec->GetFunctionArity(ENTRY_FUNCTION);
  input_names_.resize(num_inputs_);
  input_types_.resize(num_inputs_);
  input_shapes_.resize(num_inputs_);
  inputs_.resize(num_inputs_);

  try {
    for (int i = 0; i < num_inputs_; i++) {
      input_names_[i] = exec->GetFunctionParameterName(ENTRY_FUNCTION, i);
      for (auto shape : metadata_.at("Model").at("Inputs").at(i).at("shape")) {
        input_shapes_[i].push_back(shape.is_number() ? shape.get<int>() : -1);
      }
    }
    for (int i = 0; i < num_inputs_; i++) {
      input_types_[i] = metadata_.at("Model").at("Inputs").at(i).at("dtype");
    }
  } catch (nlohmann::json::out_of_range& e) {
    throw dmlc::Error(std::string("Invalid or missing input metadata: ") + e.what());
  }
}

void RelayVMModel::FetchOutputNodesData() {
  try {
    num_outputs_ = metadata_.at("Model").at("Outputs").size();
  } catch (nlohmann::json::out_of_range& e) {
    throw dmlc::Error("No output metadata found.");
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
    throw dmlc::Error("No output metadata found.");
  }
}

const char* RelayVMModel::GetInputName(int index) const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return "input";
  }
#endif

  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* RelayVMModel::GetInputType(int index) const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return "json";
  }
#endif

  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_types_[index].c_str();
}

const char* RelayVMModel::GetWeightName(int index) const { throw dmlc::Error("Not Implemented!"); }

std::vector<std::string> RelayVMModel::GetWeightNames() const {
  throw dmlc::Error("Not Implemented!");
}

void RelayVMModel::GetInput(const char* name, void* input) {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    LOG(WARNING) << "GetInput is not supported for this model.";
    return;
  }
#endif

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
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return 0;
  }
#endif

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
  if (dlr::HasNegative(input_shapes_[index].data(), input_shapes_[index].size())) return -1;
  return std::accumulate(input_shapes_[index].begin(), input_shapes_[index].end(), 1,
                         std::multiplies<int64_t>());
}

DLDataType RelayVMModel::GetInputDLDataType(int index) {
  auto input_type = input_types_[index];
  DLDataType dtype;
  dtype.lanes = 1;
  if (input_type == "bool") {
    dtype.code = kDLUInt;
    dtype.bits = 1;
  } else if (input_type == "uint8") {
    dtype.code = kDLUInt;
    dtype.bits = 8;
  } else if (input_type == "int8") {
    dtype.code = kDLInt;
    dtype.bits = 8;
  } else if (input_type == "uint16") {
    dtype.code = kDLUInt;
    dtype.bits = 16;
  } else if (input_type == "int16") {
    dtype.code = kDLInt;
    dtype.bits = 16;
  } else if (input_type == "uint32") {
    dtype.code = kDLUInt;
    dtype.bits = 32;
  } else if (input_type == "int32") {
    dtype.code = kDLInt;
    dtype.bits = 32;
  } else if (input_type == "uint64") {
    dtype.code = kDLUInt;
    dtype.bits = 64;
  } else if (input_type == "int64") {
    dtype.code = kDLInt;
    dtype.bits = 64;
  } else if (input_type == "float16") {
    dtype.code = kDLFloat;
    dtype.bits = 16;
  } else if (input_type == "bfloat16") {
    dtype.code = kDLBfloat;
    dtype.bits = 16;
  } else if (input_type == "float32") {
    dtype.code = kDLFloat;
    dtype.bits = 32;
  } else if (input_type == "float64") {
    dtype.code = kDLBfloat;
    dtype.bits = 64;
  } else {
    throw dmlc::Error(std::string("Unknown input dtype: ") + input_type);
  }
  return dtype;
}

void RelayVMModel::SetInput(const char* name, const int64_t* shape, const void* input, int dim) {
// Handle string input.
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    std::vector<DLDataType> dtypes;
    for (size_t i = 0; i < num_inputs_; ++i) {
      dtypes.emplace_back(GetInputDLDataType(i));
    }
    data_transform_.TransformInput(metadata_, shape, input, dim, dtypes, ctx_, &inputs_);
    return;
  }
#endif

  int index = GetInputIndex(name);
  DLDataType dtype = GetInputDLDataType(index);
  DLTensor input_tensor;
  input_tensor.data = const_cast<void*>(input);
  input_tensor.ctx = DLContext{DLDeviceType::kDLCPU, 0};
  input_tensor.ndim = dim;
  input_tensor.shape = const_cast<int64_t*>(shape);
  input_tensor.strides = nullptr;
  input_tensor.byte_offset = 0;
  input_tensor.dtype = dtype;
  std::vector<int64_t> arr_shape(shape, shape + dim);

  // Only allocate new buffer if not initialized or if shape or dtype has changed. Context will
  // always match.
  if (inputs_[index] == empty_ || inputs_[index].Shape() != arr_shape ||
      !TypeEqual(inputs_[index].DataType(), dtype)) {
    inputs_[index] = tvm::runtime::NDArray::Empty(arr_shape, dtype, ctx_);
  }
  inputs_[index].CopyFrom(&input_tensor);
}

void RelayVMModel::SetInputTensor(const char* name, DLTensor* tensor) {
// Handle string input.
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    std::vector<DLDataType> dtypes;
    for (size_t i = 0; i < num_inputs_; ++i) {
      dtypes.emplace_back(GetInputDLDataType(i));
    }
    data_transform_.TransformInput(metadata_, tensor->shape, tensor->data, tensor->ndim, dtypes,
                                   ctx_, &inputs_);
    return;
  }
#endif

  int index = GetInputIndex(name);
  if (index > -1) {
    std::vector<int64_t> arr_shape(tensor->shape, tensor->shape + tensor->ndim);
    tvm::runtime::NDArray input_arr = tvm::runtime::NDArray::Empty(arr_shape, tensor->dtype, ctx_);
    input_arr.CopyFrom(tensor);
    inputs_[index] = input_arr;
  }
}

void RelayVMModel::UpdateInputs() {
  const size_t num_args = num_inputs_ + 1;
  std::vector<TVMValue> values(num_args);
  std::vector<int> type_codes(num_args);
  tvm::runtime::TVMArgsSetter arg_setter(values.data(), type_codes.data());
  arg_setter(0, ENTRY_FUNCTION);
  for (int i = 0; i < inputs_.size(); i++) {
    arg_setter(i + 1, inputs_[i]);
  }

  tvm::runtime::PackedFunc set_input = vm_module_->GetFunction("set_input");
  tvm::runtime::TVMRetValue rv;
  set_input.CallPacked(tvm::runtime::TVMArgs(values.data(), type_codes.data(), num_args), &rv);
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
// Apply DataTransform if needed.
#ifdef ENABLE_DATATRANSFORM
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, i)) {
      data_transform_.TransformOutput(metadata_, i, outputs_[i]);
    }
  }
#endif
}

void RelayVMModel::GetOutput(int index, void* output) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  auto out_array = outputs_[index];
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutput(index, output);
    return;
  }
#endif

  DLTensor output_tensor;
  output_tensor.data = output;
  output_tensor.ctx = DLContext{DLDeviceType::kDLCPU, 0};
  output_tensor.ndim = out_array->ndim;
  output_tensor.dtype = out_array->dtype;
  output_tensor.shape = out_array->shape;
  output_tensor.strides = nullptr;
  output_tensor.byte_offset = 0;
  out_array.CopyTo(&output_tensor);
}

const void* RelayVMModel::GetOutputPtr(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    return data_transform_.GetOutputPtr(index);
  }
#endif

  return outputs_[index]->data;
}

void RelayVMModel::GetOutputManagedTensorPtr(int index, const DLManagedTensor** out) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  auto out_array = outputs_[index];
#ifdef ENABLE_DATATRANSFORM
  CHECK(!(HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)))
      << "Output transforms are not supported with GetOutputManagedTensor.";
#endif
  *out = out_array.ToDLPack();
}

void RelayVMModel::GetOutputTensor(int index, DLTensor* out) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  auto out_array = outputs_[index];
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutput(index, out->data);
    return;
  }
#endif
  out_array.CopyTo(out);
}

void RelayVMModel::GetOutputShape(int index, int64_t* shape) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutputShape(index, shape);
    return;
  }
#endif
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
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutputSizeDim(index, size, dim);
    return;
  }
#endif

  *size = 1;
  if (index < outputs_.size()) {
    auto arr = outputs_[index];
    if (dlr::HasNegative(arr->shape, arr->ndim)) {
      *size = -1;
    } else {
      *size = std::accumulate(arr->shape, arr->shape + arr->ndim, 1, std::multiplies<int64_t>());
    }
    *dim = arr->ndim;
  } else {
    if (dlr::HasNegative(output_shapes_[index].data(), output_shapes_[index].size())) {
      *size = -1;
    } else {
      *size = std::accumulate(output_shapes_[index].begin(), output_shapes_[index].end(), 1,
                              std::multiplies<int64_t>());
    }
    *dim = output_shapes_[index].size();
  }
}

const char* RelayVMModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    return "json";
  }
#endif

  return output_types_[index].c_str();
}

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

int RelayVMModel::GetNumInputs() const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return 1;
  }
#endif

  return num_inputs_;
}
