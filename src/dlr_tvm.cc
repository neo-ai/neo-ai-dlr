#include "dlr_tvm.h"

#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <numeric>

using namespace dlr;

ModelPath dlr::GetTvmPaths(std::vector<std::string> dirname) {
  ModelPath paths;
  std::vector<std::string> paths_vec;
  for (auto dir : dirname) {
    ListDir(dir, paths_vec);
  }
  for (auto filename : paths_vec) {
    std::string basename = GetBasename(filename);
    if (EndsWith(filename, ".json") &&
        std::all_of(
            std::begin(SAGEMAKER_AUXILIARY_JSON_FILES),
            std::end(SAGEMAKER_AUXILIARY_JSON_FILES),
            [basename](const std::string& s) { return (s != basename); }) &&
        filename != "version.json") {
      paths.model_json = filename;
    } else if (filename != LIBDLR && EndsWith(filename, LIBEXT)) {
      paths.model_lib = filename;
    } else if (EndsWith(filename, ".tensorrt")) {
      paths.model_lib = filename;
    } else if (EndsWith(filename, ".params")) {
      paths.params = filename;
    } else if (filename == "version.json") {
      paths.ver_json = filename;
    } else if (EndsWith(filename, ".meta")) {
      paths.metadata = filename;
    }
  }
  if (paths.model_json.empty() || paths.model_lib.empty() ||
      paths.params.empty()) {
    LOG(INFO) << "No valid TVM model files found under folder:";
    for (auto dir : dirname) {
      LOG(INFO) << dir;
    }
    LOG(FATAL);
  }
  return paths;
}

bool IsFileEmpty(const std::string& filePath) {
  std::ifstream pFile(filePath);
  return pFile.peek() == std::ifstream::traits_type::eof();
}

void TVMModel::SetupTVMModule(std::vector<std::string> model_path) {
  ModelPath paths = GetTvmPaths(model_path);
  std::ifstream jstream(paths.model_json);
  std::stringstream json_blob;
  json_blob << jstream.rdbuf();
  std::ifstream pstream(paths.params, std::ios::in | std::ios::binary);
  std::stringstream param_blob;
  param_blob << pstream.rdbuf();

  tvm::runtime::Module module;
  if (!IsFileEmpty(paths.model_lib)) {
    module = tvm::runtime::Module::LoadFromFile(paths.model_lib);
  }
  if (!paths.metadata.empty() && !IsFileEmpty(paths.metadata)) {
    LOG(INFO) << "Loading metadata file: " << paths.metadata;
    LoadJsonFromFile(paths.metadata, this->metadata);
  } else {
    LOG(INFO) << "No metadata found";
  }

  tvm_graph_runtime_ = tvm::runtime::make_object<tvm::runtime::GraphRuntime>();
  tvm_graph_runtime_->Init(json_blob.str(), module, {ctx_});
  tvm_graph_runtime_->LoadParams(param_blob.str());

  tvm_module_ = std::make_shared<tvm::runtime::Module>(
      tvm::runtime::Module(tvm_graph_runtime_));

  // This is the combined count of inputs and weights
  const auto num_inputs_weights = tvm_graph_runtime_->NumInputs();
  std::vector<std::string> input_names;
  for (int i = 0; i < num_inputs_weights; i++) {
    input_names.push_back(tvm_graph_runtime_->GetInputName(i));
  }
  // Get list of weights
  weight_names_ = tvm_graph_runtime_->GetWeightNames();
  num_weights_ = weight_names_.size();
  // tvm_graph_runtime_->GetInputName(*) returns both inputs and weights
  // Compute set difference to get names of inputs only
  std::sort(input_names.begin(), input_names.end());
  std::sort(weight_names_.begin(), weight_names_.end());
  std::set_difference(input_names.begin(), input_names.end(),
                      weight_names_.begin(), weight_names_.end(),
                      std::inserter(input_names_, input_names_.begin()));
  // Save the number of inputs
  num_inputs_ = input_names_.size();
  input_types_.resize(num_inputs_);
  for (int i = 0; i < num_inputs_; i++) {
    input_types_[i] = tvm_graph_runtime_->GetInputType(i);
  }

  // Get the number of output and reserve space to save output tensor
  // pointers.
  num_outputs_ = tvm_graph_runtime_->NumOutputs();
  outputs_.resize(num_outputs_);
  output_types_.resize(num_outputs_);
  for (int i = 0; i < num_outputs_; i++) {
    tvm::runtime::NDArray output = tvm_graph_runtime_->GetOutput(i);
    outputs_[i] = output.operator->();
    output_types_[i] = tvm_graph_runtime_->GetOutputType(i);
  }
}

std::vector<std::string> TVMModel::GetWeightNames() const {
  return tvm_graph_runtime_->GetWeightNames();
}

const char* TVMModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* TVMModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_types_[index].c_str();
}

const char* TVMModel::GetWeightName(int index) const {
  CHECK_LT(index, num_weights_) << "Weight index is out of range.";
  return weight_names_[index].c_str();
}

void TVMModel::SetInput(const char* name, const int64_t* shape, void* input,
                        int dim) {
  std::string str(name);
  int index = tvm_graph_runtime_->GetInputIndex(str);
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  DLTensor input_tensor = *(arr.operator->());
  input_tensor.ctx = DLContext{kDLCPU, 0};
  input_tensor.data = input;
  int64_t read_size =
      std::accumulate(shape, shape + dim, 1, std::multiplies<int64_t>());
  int64_t expected_size = std::accumulate(
      input_tensor.shape, input_tensor.shape + input_tensor.ndim, 1,
      std::multiplies<int64_t>());
  CHECK_SHAPE("Mismatch found in input data size", read_size, expected_size);
  tvm::runtime::PackedFunc set_input = tvm_module_->GetFunction("set_input");
  set_input(str, &input_tensor);
}

void TVMModel::GetInput(const char* name, void* input) {
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
  std::memcpy(shape, outputs_[index]->shape,
              sizeof(int64_t) * outputs_[index]->ndim);
}

void TVMModel::GetOutput(int index, void* out) {
  DLTensor output_tensor = *outputs_[index];
  output_tensor.ctx = DLContext{kDLCPU, 0};
  output_tensor.data = out;
  tvm::runtime::PackedFunc get_output = tvm_module_->GetFunction("get_output");
  get_output(index, &output_tensor);
}

void TVMModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  *size = 1;
  const DLTensor* tensor = outputs_[index];
  for (int i = 0; i < tensor->ndim; ++i) {
    *size *= tensor->shape[i];
  }
  *dim = tensor->ndim;
}

const char* TVMModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_types_[index].c_str();
}

void TVMModel::Run() {
  tvm::runtime::PackedFunc run = tvm_module_->GetFunction("run");
  run();
}

const char* TVMModel::GetBackend() const { return "tvm"; }

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

bool TVMModel::HasMetadata() const { return !this->metadata.is_null(); }

const char* TVMModel::GetOutputName(const int index) const {
  if (!this->HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  try {
    return this->metadata.at("Model")
        .at("Outputs")
        .at(index)
        .at("name")
        .get_ref<const std::string&>()
        .c_str();
  } catch (nlohmann::json::out_of_range& e) {
    LOG(ERROR) << e.what();
    std::string msg = "Output node with index";
    msg += " " + std::to_string(index);
    msg += " was not found in metadata file!";
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

    std::string msg = "Couldn't find index for output node";
    msg += " " + std::string{name} + "!";
    throw dmlc::Error(msg);
}

void TVMModel::GetOutputByName(const char* name, void* out) {
  int output_index = this->GetOutputIndex(name);
  this->GetOutput(output_index, out);
}
