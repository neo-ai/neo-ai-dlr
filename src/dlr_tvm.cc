#include "dlr_tvm.h"

#include <stdlib.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iterator>
#include <numeric>

using namespace dlr;

void TVMModel::SetupTVMModule(const std::vector<std::string>& files) {
  ModelPath path;
  dlr::InitModelPath(files, &path);
  if (path.model_json.empty() || path.model_lib.empty() || path.params.empty()) {
    throw dmlc::Error("Invalid TVM model artifact. Must have .so, .json, and .params files.");
  }

  std::vector<DLRModelElem> model_elems = {
      {DLRModelElemType::TVM_GRAPH, path.model_json.c_str(), nullptr, 0},
      {DLRModelElemType::TVM_PARAMS, path.params.c_str(), nullptr, 0},
      {DLRModelElemType::TVM_LIB, path.model_lib.c_str(), nullptr, 0}};
  if (!path.metadata.empty()) {
    model_elems.push_back({DLRModelElemType::NEO_METADATA, path.metadata.c_str(), nullptr, 0});
  }
  SetupTVMModule(model_elems);
}

void TVMModel::SetupTVMModule(const std::vector<DLRModelElem>& model_elems) {
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

  std::string graph_str;
  DLRString params_str;
  const char* params_data = nullptr;
  size_t params_size = 0;
  std::string model_lib_path;
  std::string metadata_data;
  for (DLRModelElem el : model_elems) {
    if (el.type == DLRModelElemType::TVM_GRAPH) {
      if (el.path != nullptr) {
        graph_str = dlr::LoadFileToString(el.path);
      } else if (el.data != nullptr) {
        graph_str = static_cast<const char*>(el.data);
      } else {
        throw dmlc::Error("Invalid TVM model element TVM_GRAPH");
      }
    } else if (el.type == DLRModelElemType::TVM_PARAMS) {
      if (el.path != nullptr) {
        std::ifstream pstream(el.path, std::ios::in | std::ios::binary);
        DLRStringStream params_blob;
        params_blob << pstream.rdbuf();
        params_str = params_blob.str();
        params_data = params_str.data();
        params_size = params_str.size();
      } else if (el.data != nullptr && el.data_size > 0) {
        params_data = static_cast<const char*>(el.data);
        params_size = el.data_size;
      } else {
        throw dmlc::Error("Invalid TVM model element TVM_PARAMS");
      }
    } else if (el.type == DLRModelElemType::TVM_LIB) {
      if (el.path != nullptr) {
        model_lib_path = el.path;
      } else {
        throw dmlc::Error("Invalid TVM model element TVM_LIB. TVM_LIB must be a file path.");
      }
    } else if (el.type == DLRModelElemType::NEO_METADATA) {
      if (el.path != nullptr) {
        metadata_data = dlr::LoadFileToString(el.path);
      } else if (el.data != nullptr) {
        metadata_data = static_cast<const char*>(el.data);
      }
    }
  }
  if (graph_str.empty() || params_data == nullptr || params_size <= 0 || model_lib_path.empty()) {
    throw dmlc::Error("Invalid TVM model. Must have TVM_GRAPH, TVM_PARAMS and TVM_LIB elements");
  }
  if (!metadata_data.empty()) {
    LoadJsonFromString(metadata_data, this->metadata_);
    ValidateDeviceTypeIfExists();
  }

  tvm::runtime::Module module;
  module = tvm::runtime::Module::LoadFromFile(model_lib_path);

  tvm_graph_runtime_ = tvm::runtime::make_object<tvm::runtime::GraphRuntime>();
  tvm_graph_runtime_->Init(graph_str, module, {ctx_}, nullptr);
  dmlc::MemoryFixedSizeStream strm(const_cast<char*>(params_data), params_size);
  tvm_graph_runtime_->LoadParams(&strm);

  tvm_module_ = std::make_shared<tvm::runtime::Module>(tvm::runtime::Module(tvm_graph_runtime_));

  // Get list of weights.
  weight_names_ = tvm_graph_runtime_->GetWeightNames();
  num_weights_ = weight_names_.size();
  std::unordered_set<std::string> weight_names_set(weight_names_.begin(), weight_names_.end());
  // TVM inputs contains both inputs and weights.
  const auto num_inputs_weights = tvm_graph_runtime_->NumInputs();
  // Filter out weights to get only inputs.
  for (int i = 0; i < num_inputs_weights; i++) {
    auto name = tvm_graph_runtime_->GetInputName(i);
    if (weight_names_set.count(name) == 0) {
      input_names_.push_back(name);
    }
  }
  // Save the number of inputs
  num_inputs_ = input_names_.size();
  inputs_.resize(num_inputs_);
  input_types_.resize(num_inputs_);
  for (int i = 0; i < num_inputs_; i++) {
    inputs_[i] = tvm_graph_runtime_->GetInput(i);
    input_types_[i] = tvm_graph_runtime_->GetInputType(i);
  }

  // Get the number of output and reserve space to save output tensor
  // pointers.
  num_outputs_ = tvm_graph_runtime_->NumOutputs();
  outputs_.resize(num_outputs_);
  output_types_.resize(num_outputs_);
  for (int i = 0; i < num_outputs_; i++) {
    outputs_[i] = tvm_graph_runtime_->GetOutput(i);
    output_types_[i] = tvm_graph_runtime_->GetOutputType(i);
  }
  UpdateInputShapes();
}

void TVMModel::UpdateInputShapes() {
  input_shapes_.resize(num_inputs_);
  for (int i = 0; i < num_inputs_; i++) {
    std::vector<int64_t> input_shape;
    tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(i);
    input_shape.assign(arr->shape, arr->shape + arr->ndim);
    input_shapes_[i] = input_shape;
  }
}

std::vector<std::string> TVMModel::GetWeightNames() const { return weight_names_; }

const char* TVMModel::GetInputName(int index) const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return "input";
  }
#endif

  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* TVMModel::GetInputType(int index) const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return "json";
  }
#endif

  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_types_[index].c_str();
}

const int TVMModel::GetInputDim(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  return arr->ndim;
}

const int64_t TVMModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  if (dlr::HasNegative(arr->shape, arr->ndim)) return -1;
  return std::accumulate(arr->shape, arr->shape + arr->ndim, 1, std::multiplies<int64_t>());
}

const char* TVMModel::GetWeightName(int index) const {
  CHECK_LT(index, num_weights_) << "Weight index is out of range.";
  return weight_names_[index].c_str();
}

void TVMModel::SetInput(const char* name, const int64_t* shape, const void* input, int dim) {
#ifdef ENABLE_DATATRANSFORM
  // Handle string input.
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    std::vector<DLDataType> dtypes;
    for (size_t i = 0; i < num_inputs_; ++i) {
      dtypes.emplace_back(inputs_[i]->dtype);
    }
    data_transform_.TransformInput(metadata_, shape, input, dim, dtypes, ctx_, &inputs_);
    return;
  }
#endif

  std::string str(name);
  int index = tvm_graph_runtime_->GetInputIndex(str);
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  DLTensor input_tensor = *(arr.operator->());
  input_tensor.ctx = DLContext{kDLCPU, 0};
  input_tensor.data = const_cast<void*>(input);
  int64_t read_size = std::accumulate(shape, shape + dim, 1, std::multiplies<int64_t>());
  int64_t expected_size = std::accumulate(
      input_tensor.shape, input_tensor.shape + input_tensor.ndim, 1, std::multiplies<int64_t>());
  CHECK_SHAPE("Mismatch found in input data size", read_size, expected_size);
  tvm::runtime::PackedFunc set_input = tvm_module_->GetFunction("set_input");
  set_input(str, &input_tensor);
  UpdateInputShapes();
}

void TVMModel::SetInputTensor(const char* name, DLTensor* tensor) {
#ifdef ENABLE_DATATRANSFORM
  // Handle string input.
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    std::vector<DLDataType> dtypes;
    for (size_t i = 0; i < num_inputs_; ++i) {
      dtypes.emplace_back(inputs_[i]->dtype);
    }
    data_transform_.TransformInput(metadata_, tensor->shape, tensor->data, tensor->ndim, dtypes,
                                   ctx_, &inputs_);
    return;
  }
#endif

  std::string str(name);
  int index = tvm_graph_runtime_->GetInputIndex(str);
  if (index > -1) {
    tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
    DLTensor input_tensor = *(arr.operator->());
    int64_t read_size =
        std::accumulate(tensor->shape, tensor->shape + tensor->ndim, 1, std::multiplies<int64_t>());
    int64_t expected_size = std::accumulate(
        input_tensor.shape, input_tensor.shape + input_tensor.ndim, 1, std::multiplies<int64_t>());
    CHECK_SHAPE("Mismatch found in input data size", read_size, expected_size);
    tvm_graph_runtime_->SetInput(index, tensor);
  }
}

void TVMModel::SetInputTensorZeroCopy(const char* name, DLTensor* tensor) {
  std::string str(name);
  int index = tvm_graph_runtime_->GetInputIndex(str);
  if (index == -1) return;
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  const DLTensor* old_t = arr.operator->();
  CHECK_EQ(reinterpret_cast<size_t>(tensor->data) % 128, 0)
      << "Data must be alligned to 128 bits for SetDLRInputTensorZeroCopy.";
  CHECK_EQ(old_t->ndim, static_cast<size_t>(tensor->ndim))
      << "Model expected " << old_t->ndim << " dimensions, but input has " << tensor->ndim;
  CHECK_EQ(old_t->ctx.device_type, tensor->ctx.device_type)
      << "The input data must be on device \"" << GetStringFromDeviceType(old_t->ctx.device_type)
      << "\", but user gave input on \"" << GetStringFromDeviceType(tensor->ctx.device_type)
      << "\"";
  CHECK_EQ(old_t->ctx.device_id, tensor->ctx.device_id);
  for (auto i = 0; i < tensor->ndim; ++i) {
    CHECK_EQ(old_t->shape[i], tensor->shape[i]);
  }
  tvm_graph_runtime_->SetInputZeroCopy(index, tensor);
}

void TVMModel::GetInput(const char* name, void* input) {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    LOG(WARNING) << "GetInput is not supported for this model.";
    return;
  }
#endif

  std::string str(name);
  int index = tvm_graph_runtime_->GetInputIndex(str);
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  DLTensor input_tensor;
  input_tensor.data = input;
  input_tensor.ctx = DLContext{kDLCPU, 0};
  input_tensor.ndim = arr->ndim;
  input_tensor.dtype = arr->dtype;
  input_tensor.shape = arr->shape;
  input_tensor.strides = nullptr;
  input_tensor.byte_offset = 0;
  arr.CopyTo(&input_tensor);
}

void TVMModel::GetOutputShape(int index, int64_t* shape) const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutputShape(index, shape);
    return;
  }
#endif
  std::memcpy(shape, outputs_[index]->shape, sizeof(int64_t) * outputs_[index]->ndim);
}

void TVMModel::GetOutput(int index, void* out) {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutput(index, out);
    return;
  }
#endif
  DLTensor output_tensor = *outputs_[index].operator->();
  output_tensor.ctx = DLContext{kDLCPU, 0};
  output_tensor.data = out;
  tvm::runtime::PackedFunc get_output = tvm_module_->GetFunction("get_output");
  get_output(index, &output_tensor);
}

const void* TVMModel::GetOutputPtr(int index) const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    return data_transform_.GetOutputPtr(index);
  }
#endif

  tvm::runtime::NDArray output = tvm_graph_runtime_->GetOutput(index);
  const DLTensor* tensor = output.operator->();
  if (tensor->ctx.device_type == kDLCPU) {
    return tensor->data;
  }
  throw dmlc::Error("GetOutputPtr is not supported for non-CPU device types");
}

void TVMModel::GetOutputManagedTensorPtr(int index, const DLManagedTensor** out) {
#ifdef ENABLE_DATATRANSFORM
  CHECK(!(HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)))
      << "Output transforms are not supported with GetOutputManagedTensor.";
#endif
  tvm::runtime::NDArray output = tvm_graph_runtime_->GetOutput(index);
  *out = output.ToDLPack();
}

void TVMModel::GetOutputTensor(int index, DLTensor* out) {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutput(index, out->data);
    return;
  }
#endif
  tvm::runtime::PackedFunc get_output = tvm_module_->GetFunction("get_output");
  get_output(index, out);
}

void TVMModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    data_transform_.GetOutputSizeDim(index, size, dim);
    return;
  }
#endif
  *size = 1;
  const DLTensor* tensor = outputs_[index].operator->();
  for (int i = 0; i < tensor->ndim; ++i) {
    if (tensor->shape[i] < 0) {
      *size = -1;
      break;
    }
    *size *= tensor->shape[i];
  }
  *dim = tensor->ndim;
}

const char* TVMModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, index)) {
    return "json";
  }
#endif

  return output_types_[index].c_str();
}

void TVMModel::Run() {
  tvm::runtime::PackedFunc run = tvm_module_->GetFunction("run");
  run();
#ifdef ENABLE_DATATRANSFORM
  // Apply DataTransform if needed.
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (HasMetadata() && data_transform_.HasOutputTransform(metadata_, i)) {
      data_transform_.TransformOutput(metadata_, i, outputs_[i]);
    }
  }
#endif
}

static inline int SetEnv(const char* key, const char* value) {
#ifdef _WIN32
  return static_cast<int>(_putenv_s(key, value));
#else
  return setenv(key, value, 1);
#endif  // _WIN32
}

void TVMModel::SetNumThreads(int threads) {
  if (threads > 0) {
    SetEnv("TVM_NUM_THREADS", std::to_string(threads).c_str());
    LOG(INFO) << "Set Num Threads: " << threads;
  }
}

void TVMModel::UseCPUAffinity(bool use) {
  if (use) {
    SetEnv("TVM_BIND_THREADS", "1");
    LOG(INFO) << "CPU Affinity is enabled";
  } else {
    SetEnv("TVM_BIND_THREADS", "0");
    LOG(INFO) << "CPU Affinity is disabled";
  }
}

const char* TVMModel::GetOutputName(const int index) const {
  if (!this->HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  try {
    return this->metadata_.at("Model")
        .at("Outputs")
        .at(index)
        .at("name")
        .get_ref<const std::string&>()
        .c_str();
  } catch (nlohmann::json::out_of_range& e) {
    std::string msg =
        "Output node with index " + std::to_string(index) + " was not found in metadata file!";
    throw dmlc::Error(msg);
  }
}

int TVMModel::GetOutputIndex(const char* name) const {
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

  std::string msg = "Couldn't find index for output node " + std::string(name) + "!";
  throw dmlc::Error(msg);
}

void TVMModel::GetOutputByName(const char* name, void* out) {
  int output_index = this->GetOutputIndex(name);
  this->GetOutput(output_index, out);
}

int TVMModel::GetNumInputs() const {
#ifdef ENABLE_DATATRANSFORM
  if (HasMetadata() && data_transform_.HasInputTransform(metadata_)) {
    return 1;
  }
#endif

  return num_inputs_;
}
