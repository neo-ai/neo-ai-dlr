#include "dlr_tvm.h"

#include <stdlib.h>

#include <fstream>
#include <iterator>
#include <numeric>

using namespace dlr;

void TVMModel::InitModelArtifact() {
  std::shared_ptr<TVMModelArtifact> artifact =
      std::make_shared<TVMModelArtifact>();
  std::vector<std::string> filenames = ListFilesInDirectories(paths_);
  for (auto filename : filenames) {
    std::string basename = GetBasename(filename);
    if (EndsWith(filename, ".json") &&
        std::all_of(
            std::begin(SAGEMAKER_AUXILIARY_JSON_FILES),
            std::end(SAGEMAKER_AUXILIARY_JSON_FILES),
            [basename](const std::string& s) { return (s != basename); }) &&
        filename != "version.json") {
      artifact->model_json = filename;
    } else if (filename != LIBDLR && EndsWith(filename, LIBEXT)) {
      artifact->model_lib = filename;
    } else if (EndsWith(filename, ".tensorrt")) {
      artifact->model_lib = filename;
    } else if (EndsWith(filename, ".params")) {
      artifact->params = filename;
    } else if (filename == "version.json") {
      artifact->ver_json = filename;
    } else if (EndsWith(filename, ".meta")) {
      artifact->metadata = filename;
    }
  }
  if (artifact->model_json.empty() || artifact->model_lib.empty() ||
      artifact->params.empty()) {
    LOG(INFO) << "No valid TVM model files found under folder:";
    for (auto dir : paths_) {
      LOG(INFO) << dir;
    }
    LOG(FATAL);
  } else {
    model_artifact_ = std::shared_ptr<ModelArtifact>(artifact);
  }
}

void TVMModel::SetupTvmGraphRuntime() {
  std::shared_ptr<TVMModelArtifact> artifact =
      std::static_pointer_cast<TVMModelArtifact>(model_artifact_);
  std::ifstream jstream(artifact->model_json);
  std::stringstream json_blob;
  json_blob << jstream.rdbuf();
  std::ifstream pstream(artifact->params, std::ios::in | std::ios::binary);
  std::stringstream param_blob;
  param_blob << pstream.rdbuf();

  tvm::runtime::Module module;
  if (!IsFileEmpty(artifact->model_lib)) {
    module = tvm::runtime::Module::LoadFromFile(artifact->model_lib);
  }

  tvm_graph_runtime_ = tvm::runtime::make_object<tvm::runtime::GraphRuntime>();
  tvm_graph_runtime_->Init(json_blob.str(), module, {ctx_});
  tvm_graph_runtime_->LoadParams(param_blob.str());
}

void TVMModel::FetchInputNodesData() {
  // This is the combined count of inputs and weights
  const auto num_input_and_weight_nodes = tvm_graph_runtime_->NumInputs();
  std::vector<std::string> input_and_weight_node_names;
  for (int i = 0; i < num_input_and_weight_nodes; i++) {
    input_and_weight_node_names.push_back(tvm_graph_runtime_->GetInputName(i));
  }
  // Get list of weights
  std::vector<std::string> weight_names = tvm_graph_runtime_->GetWeightNames();
  // tvm_graph_runtime_.GetInputName(*) returns both inputs and weights
  // Compute set difference to get names of inputs only
  std::sort(input_and_weight_node_names.begin(),
            input_and_weight_node_names.end());
  std::sort(weight_names.begin(), weight_names.end());
  std::set_difference(input_and_weight_node_names.begin(),
                      input_and_weight_node_names.end(), weight_names.begin(),
                      weight_names.end(),
                      std::inserter(input_names_, input_names_.begin()));
  // Save the number of inputs
  num_inputs_ = input_names_.size();
  input_types_.resize(num_inputs_);
  for (int i = 0; i < num_inputs_; i++) {
    int input_index = tvm_graph_runtime_->GetInputIndex(input_names_[i]);
    input_types_[i] = tvm_graph_runtime_->GetInputType(input_index);
  }

  UpdateInputShapes();
}

void TVMModel::FetchOutputNodesData() {
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
  UpdateOutputShapes();
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

void TVMModel::UpdateOutputShapes() {
  output_shapes_.resize(num_outputs_);
  for (int i = 0; i < num_outputs_; i++) {
    std::vector<int64_t> output_shape;
    output_shape.assign(outputs_[i]->shape,
                        outputs_[i]->shape + outputs_[i]->ndim);
    output_shapes_[i] = output_shape;
  }
}

const int TVMModel::GetInputDim(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  return arr->ndim;
}

const int64_t TVMModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  return std::accumulate(arr->shape, arr->shape + arr->ndim, 1,
                         std::multiplies<int64_t>());
}

void TVMModel::GetInput(int index, void* input) {
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

void TVMModel::GetInput(const char* name, void* input) {
  std::string node_name(name);
  int index = tvm_graph_runtime_->GetInputIndex(node_name);
  GetInput(index, input);
}

void TVMModel::SetInput(const char* name, const int64_t* shape, void* input,
                        int dim) {
  std::string node_name(name);
  int index = tvm_graph_runtime_->GetInputIndex(node_name);
  int64_t read_size =
      std::accumulate(shape, shape + dim, 1, std::multiplies<int64_t>());
  CHECK_SHAPE("Mismatch found in input data size", read_size,
              GetInputSize(index));
  SetInput(node_name, *shape, input);
}

void TVMModel::SetInput(const int index, const int64_t batch_size,
                        void* input) {
  tvm::runtime::NDArray arr = tvm_graph_runtime_->GetInput(index);
  DLTensor input_tensor = *(arr.operator->());
  input_tensor.ctx = DLContext{kDLCPU, 0};
  input_tensor.data = input;
  tvm_graph_runtime_->SetInput(index, &input_tensor);

  // Updated input and output shapes to account for batch size.
  UpdateInputShapes();
  UpdateOutputShapes();
}

void TVMModel::SetInput(std::string name, const int64_t batch_size,
                        void* input) {
  int index = tvm_graph_runtime_->GetInputIndex(name);
  SetInput(index, batch_size, input);
}

const int TVMModel::GetOutputDim(int index) const {
  return output_shapes_[index].size();
}

const int64_t TVMModel::GetOutputSize(int index) const {
  return std::accumulate(output_shapes_[index].begin(),
                         output_shapes_[index].end(), 1,
                         std::multiplies<int64_t>());
}

void TVMModel::GetOutput(int index, void* out) {
  tvm::runtime::NDArray output = tvm_graph_runtime_->GetOutput(index);
  DLManagedTensor* output_tensor = output.ToDLPack();
  std::memcpy(out, output_tensor->dl_tensor.data, GetOutputSize(index));
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

void TVMModel::Run() { tvm_graph_runtime_->Run(); }
