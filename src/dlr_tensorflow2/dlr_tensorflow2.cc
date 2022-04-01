#include "dlr_tensorflow2/dlr_tensorflow2.h"

#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>

using namespace dlr;

void dlr::PrepareTF2ConfigProto(const DLR_TF2Config& tf2_config,
                                std::vector<std::uint8_t>& config) {
  if (tf2_config.intra_op_parallelism_threads > 0 &&
      tf2_config.intra_op_parallelism_threads < 256) {
    // More info https://github.com/tensorflow/tensorflow/issues/13853
    config.insert(config.end(), {0x10, (std::uint8_t)tf2_config.intra_op_parallelism_threads});
    LOG(INFO) << "Set intra_op_parallelism_threads to " << tf2_config.intra_op_parallelism_threads;
  }
  if (tf2_config.inter_op_parallelism_threads > 0 &&
      tf2_config.inter_op_parallelism_threads < 256) {
    // More info https://github.com/tensorflow/tensorflow/issues/13853
    config.insert(config.end(), {0x28, (std::uint8_t)tf2_config.inter_op_parallelism_threads});
    LOG(INFO) << "Set inter_op_parallelism_threads to " << tf2_config.inter_op_parallelism_threads;
  }

  // Tensorflow GPUOptions
  std::vector<std::uint8_t> gpu_options = {0x32, 0x0};

  double gpu_memory_fraction = tf2_config.gpu_options.per_process_gpu_memory_fraction;
  if (gpu_memory_fraction > 0.001 && gpu_memory_fraction < 1.0) {
    std::uint8_t proto[9] = {0x9, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    auto bytes = reinterpret_cast<std::uint8_t*>(&gpu_memory_fraction);
    // Put it to the config byte-array, from 1 to 9:
    for (std::size_t i = 0; i < sizeof(gpu_memory_fraction); i++) {
      proto[i + 1] = bytes[i];
    }
    gpu_options.insert(gpu_options.end(), proto, proto + 9);
    LOG(INFO) << "Set gpu_options.per_process_gpu_memory_fraction to " << gpu_memory_fraction;
  }

  if (tf2_config.gpu_options.allow_growth != 0) {
    gpu_options.insert(gpu_options.end(), {0x20, 0x1});
    LOG(INFO) << "Set gpu_options.allow_growth to True";
  }
  // update gpu_options data length (stored in byte #2)
  gpu_options[1] = gpu_options.size() - 2;

  if (gpu_options[1] > 0) {
    config.insert(config.end(), gpu_options.begin(), gpu_options.end());
  }
}

TF_Output Tensorflow2Model::ParseTensorName(const std::string& t_name) {
  std::regex r("^(.+):(\\d+)$");
  std::smatch match;
  if (!std::regex_search(t_name, match, r)) {
    LOG(FATAL) << "ERROR: failed to parse tensor name " << t_name;
  }
  std::string op_name = match.str(1);
  int op_out_id = std::stoi(match.str(2));
  TF_Operation* op = TF_GraphOperationByName(graph_, op_name.c_str());
  if (op == nullptr) {
    LOG(FATAL) << "ERROR: TF_GraphOperationByName failed for operation " << op_name;
  }
  const TF_Output oper_out = {op, op_out_id};
  return oper_out;
}

void Tensorflow2Model::DetectInputs() {
  size_t pos = 0;
  TF_Operation* op;
  while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
    const std::string op_type = TF_OperationOpType(op);
    const int n_in = TF_OperationNumInputs(op);
    const int n_out = TF_OperationNumOutputs(op);
    const std::string op_name = TF_OperationName(op);
    if (op_type == "Placeholder" && n_in == 0 && n_out == 1 && ignored_names_.count(op_name) == 0) {
      input_names_.push_back(op_name + ":0");
    }
  }
  num_inputs_ = input_names_.size();
  std::string msg = "Found " + std::to_string(num_inputs_) + " possible inputs: ";
  for (int i = 0; i < num_inputs_; i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += input_names_[i];
  }
  LOG(INFO) << msg;
}

void Tensorflow2Model::DetectOutputs() {
  size_t pos = 0;
  TF_Operation* op;
  // while loop
  while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
    const std::string op_type = TF_OperationOpType(op);
    const int n_out = TF_OperationNumOutputs(op);
    const int n_cout = TF_OperationNumControlOutputs(op);
    const std::string op_name = TF_OperationName(op);
    if (op_type != "Const" && op_type != "Assign" && op_type != "NoOp" &&
        op_type != "Placeholder" && n_cout == 0 && ignored_names_.count(op_name) == 0) {
      int n_consumers = 0;
      for (int i = 0; i < n_out; i++) {
        const TF_Output tf_out = {op, i};
        n_consumers += TF_OperationOutputNumConsumers(tf_out);
        if (n_consumers != 0) {
          break;
        }
      }
      if (n_consumers != 0) {
        continue;  // while loop
      }
      for (int i = 0; i < n_out; i++) {
        output_names_.push_back(op_name + ":" + std::to_string(i));
      }
    }
  }
  num_outputs_ = output_names_.size();
  std::string msg = "Found " + std::to_string(num_outputs_) + " possible outputs: ";
  for (int i = 0; i < num_outputs_; i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += output_names_[i];
  }
  LOG(INFO) << msg;
}

void Tensorflow2Model::DetectInputShapes() {
  for (int i = 0; i < num_inputs_; i++) {
    const std::string& t_name = input_names_[i];
    const TF_Output oper_out = ParseTensorName(t_name);

    int n_dim = TF_GraphGetTensorNumDims(graph_, oper_out, status_);
    if (TF_GetCode(status_) != TF_OK) {
      LOG(FATAL) << "ERROR: TF_GraphGetTensorNumDims failed " << TF_Message(status_);
      return;  // unreachable
    }
    int64_t dims[n_dim];
    TF_GraphGetTensorShape(graph_, oper_out, dims, n_dim, status_);
    if (TF_GetCode(status_) != TF_OK) {
      LOG(FATAL) << "ERROR: TF_GraphGetTensorShape failed " << TF_Message(status_);
      return;  // unreachable
    }
    graph_input_shapes_.push_back(std::vector<int64_t>(dims, dims + n_dim));
  }
  input_shapes_.resize(num_inputs_);
}

TF_Tensor* Tensorflow2Model::AllocateInputTensor(int index, const int64_t* dims, const int n_dim) {
  const TF_Output oper_out = inputs_[index];
  size_t num_elements = 1;
  for (int z = 0; z < n_dim; z++) {
    if (dims[z] < 1) {
      LOG(FATAL) << "ERROR: non-positive dimensions are not supported. "
                 << "input id: " << index << ", dims[" << z << "]=" << dims[z];
    }
    num_elements *= dims[z];
  }
  const TF_DataType t_type = TF_OperationOutputType(oper_out);
  const size_t num_bytes = TF_DataTypeSize(t_type) * num_elements;
  TF_Tensor* tensor = TF_AllocateTensor(t_type, dims, n_dim, num_bytes);
  LOG(INFO) << "Input Tensor " << index << " was allocated";
  return tensor;
}

void Tensorflow2Model::PrepInputs() {
  for (std::string& t_name : input_names_) {
    TF_Output oper_out = ParseTensorName(t_name);
    const TF_DataType t_type = TF_OperationOutputType(oper_out);
    input_types_.push_back(std::to_string((int)t_type));
    inputs_.push_back(oper_out);
    // fill output_tensors_ vector with nulls
    input_tensors_.push_back(nullptr);
  }
}

void Tensorflow2Model::PrepOutputs() {
  for (std::string& t_name : output_names_) {
    TF_Output oper_out = ParseTensorName(t_name);
    const TF_DataType t_type = TF_OperationOutputType(oper_out);
    output_types_.push_back(std::to_string((int)t_type));
    outputs_.push_back(oper_out);
    // fill output_tensors_ vector with nulls
    output_tensors_.push_back(nullptr);
  }
}

int Tensorflow2Model::GetInputId(const char* name) {
  // In most of the cases it will be just 1 element in the vector.
  // Scan vector to find tensor by name.
  for (int i = 0; i < num_inputs_; i++) {
    if (input_names_[i].compare(name) == 0) {
      return i;
    }
  }
  LOG(FATAL) << "Input Tensor not found, name: " << name;
  return -1;  // unreachable
}

// Constructor
Tensorflow2Model::Tensorflow2Model(const std::string& model_path, const DLDevice& dev,
                                   const DLR_TF2Config& tf2_config)
    : DLRModel(dev, DLRBackend::kTENSORFLOW2) {
  status_ = TF_NewStatus();
  graph_ = TF_NewGraph();
  TF_SessionOptions* sess_opts = TF_NewSessionOptions();
  std::vector<std::uint8_t> config;
  PrepareTF2ConfigProto(tf2_config, config);
  if (!config.empty()) {
    TF_SetConfig(sess_opts, config.data(), config.size(), status_);
    if (TF_GetCode(status_) != TF_OK) {
      TF_DeleteSessionOptions(sess_opts);
      LOG(FATAL) << "ERROR: TF_SetConfig failed " << TF_Message(status_);
      return;  // unreachable
    }
  }
  TF_Buffer* run_opts = nullptr;
  TF_Buffer* meta_graph_def = nullptr;
  const char* tags = "serve";
  int ntags = 1;
  sess_ = TF_LoadSessionFromSavedModel(sess_opts, run_opts, model_path.c_str(), &tags, ntags,
                                       graph_, meta_graph_def, status_);
  if (TF_GetCode(status_) != TF_OK) {
    LOG(FATAL) << "ERROR: Unable to create Session " << TF_Message(status_);
    return;  // unreachable
  }

  auto metadata = GetMetadataFile(model_path);
  if (!metadata.empty() && !IsFileEmpty(metadata)) {
    LOG(INFO) << "Loading metadata file: " << metadata;
    LoadJsonFromFile(metadata, this->metadata_);
    LOG(INFO) << "Input and Output names from metadata file";
    LOG(INFO) << "Input Names:";
    for (auto& el : this->metadata_.at("Model").at("Inputs")) {
      input_names_.push_back(el.at("name"));
      LOG(INFO) << el.at("name");
    }
    LOG(INFO) << "Output Names:";
    for (auto& el : this->metadata_.at("Model").at("Outputs")) {
      output_names_.push_back(el.at("name"));
      LOG(INFO) << el.at("name");
    }
    num_inputs_ = input_names_.size();
    num_outputs_ = output_names_.size();
  } else {
    ignored_names_ = {
        "saver_filename",             // name of the checkpoint
        "StatefulPartitionedCall_1",  // the loss
        "StatefulPartitionedCall_2"   // save operation
    };
    LOG(WARNING) << "Metadata file was not found. Auto-detecting Input and Output names. This may "
                    "not work correctly for some models...";
    DetectInputs();
    DetectOutputs();
  }
  DetectInputShapes();
  PrepInputs();
  PrepOutputs();

  TF_DeleteSessionOptions(sess_opts);
  LOG(INFO) << "Tensorflow Session was created";
}

// Destructor
Tensorflow2Model::~Tensorflow2Model() {
  for (TF_Tensor* tensor : input_tensors_) {
    TF_DeleteTensor(tensor);
  }
  for (TF_Tensor* tensor : output_tensors_) {
    TF_DeleteTensor(tensor);
  }
  TF_CloseSession(sess_, status_);
  // Result of close is ignored, delete anyway.
  TF_DeleteSession(sess_, status_);
  TF_DeleteGraph(graph_);
  TF_DeleteStatus(status_);

  LOG(INFO) << "Tensorflow2Model was deleted";
}

std::vector<std::string> Tensorflow2Model::GetWeightNames() const {
  LOG(FATAL) << "GetWeightNames is not supported by Tensorflow backend";
  return std::vector<std::string>();  // unreachable
}

const char* Tensorflow2Model::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* Tensorflow2Model::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_types_[index].c_str();
}

const std::vector<int64_t>& Tensorflow2Model::GetInputShape(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index].empty() ? graph_input_shapes_[index] : input_shapes_[index];
}

const int Tensorflow2Model::GetInputDim(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return graph_input_shapes_[index].size();
}

const int64_t Tensorflow2Model::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  const std::vector<int64_t>& shape =
      input_shapes_[index].empty() ? graph_input_shapes_[index] : input_shapes_[index];
  if (dlr::HasNegative(shape.data(), shape.size())) return -1;
  return abs(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
}

const char* Tensorflow2Model::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Tensorflow backend";
  return "";  // unreachable
}

void Tensorflow2Model::SetInput(const char* name, const int64_t* shape, const void* input,
                                int dim) {
  int index = GetInputId(name);
  TF_Tensor* tensor = input_tensors_[index];

  bool keep = tensor != nullptr && TF_NumDims(tensor) == dim;
  if (keep) {
    for (int i = 0; i < dim; i++) {
      if (shape[i] != TF_Dim(tensor, i)) {
        keep = false;
        break;
      }
    }
  }
  if (!keep) {
    if (tensor != nullptr) {
      TF_DeleteTensor(tensor);
      input_tensors_[index] = nullptr;
    }
    tensor = AllocateInputTensor(index, shape, dim);
    input_tensors_[index] = tensor;
    input_shapes_[index] = std::vector<int64_t>(shape, shape + dim);
  }
  size_t num_bytes = TF_TensorByteSize(tensor);
  void* in_t_data = TF_TensorData(tensor);
  std::memcpy(in_t_data, input, num_bytes);
}

void Tensorflow2Model::GetInput(const char* name, void* input) {
  int index = GetInputId(name);
  TF_Tensor* tensor = input_tensors_[index];
  CHECK_NOTNULL(tensor);
  size_t num_bytes = TF_TensorByteSize(tensor);
  void* in_t_data = TF_TensorData(tensor);
  std::memcpy(input, in_t_data, num_bytes);
}

const char* Tensorflow2Model::GetOutputName(const int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_names_[index].c_str();
}

void Tensorflow2Model::GetOutputShape(int index, int64_t* shape) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  CHECK_NOTNULL(tensor);
  int n_dim = TF_NumDims(tensor);
  for (int i = 0; i < n_dim; i++) {
    shape[i] = TF_Dim(tensor, i);
  }
}

void Tensorflow2Model::GetOutput(int index, void* output) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  const TF_Tensor* tensor = output_tensors_[index];
  CHECK_NOTNULL(tensor);
  size_t num_bytes = TF_TensorByteSize(tensor);
  const void* out_t_data = TF_TensorData(tensor);
  std::memcpy(output, out_t_data, num_bytes);
}

const void* Tensorflow2Model::GetOutputPtr(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  const TF_Tensor* tensor = output_tensors_[index];
  CHECK_NOTNULL(tensor);
  const void* out_t_data = TF_TensorData(tensor);
  return out_t_data;
}

void Tensorflow2Model::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  CHECK_NOTNULL(tensor);
  *size = TF_TensorElementCount(tensor);
  *dim = TF_NumDims(tensor);
}

const char* Tensorflow2Model::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_types_[index].c_str();
}

void Tensorflow2Model::Run() {
  // Delete previous output Tensors to prevent GPU memory leak
  for (TF_Tensor* tensor : output_tensors_) {
    if (tensor != nullptr) {
      TF_DeleteTensor(tensor);
    }
  }
  TF_SessionRun(sess_,
                nullptr,  // Run options.
                inputs_.data(), input_tensors_.data(), num_inputs_, outputs_.data(),
                output_tensors_.data(), num_outputs_, nullptr,
                0,        // Target operations, number of targets.
                nullptr,  // Run metadata.
                status_   // Output status.
  );
  if (TF_GetCode(status_) != TF_OK) {
    LOG(FATAL) << "ERROR: Session run error: " << TF_Message(status_);
    return;  // unreachable
  }
}

void Tensorflow2Model::SetNumThreads(int threads) {
  LOG(FATAL) << "SetNumThreads is not supported by Tensorflow backend";
}

void Tensorflow2Model::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by Tensorflow backend";
}
