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

void Tensorflow2Model::DetectInputsAndOutputs(const InputOutputType& inputs,
                                              const InputOutputType& outputs) {
  for (auto& el : inputs) {
    const tensorflow::TensorInfo& ti = el.second;
    input_names_.push_back(el.first);
    input_tensor_names_.push_back(ti.name());
    input_types_.push_back(DataType_Name(ti.dtype()));
    const tensorflow::TensorShapeProto& shape = ti.tensor_shape();
    int dim_size = shape.dim_size();
    std::vector<int64_t> dims;
    for (int i = 0; i < dim_size; i++) {
      const tensorflow::TensorShapeProto_Dim& dim = shape.dim(i);
      int64_t dim_sz = dim.size();
      dims.push_back(dim_sz);
    }
    graph_input_shapes_.push_back(dims);
    TF_Output oper_out = ParseTensorName(ti.name());
    inputs_.push_back(oper_out);
    // fill output_tensors_ vector with nulls
    input_tensors_.push_back(nullptr);
  }
  for (auto& el : outputs) {
    const tensorflow::TensorInfo& ti = el.second;
    output_names_.push_back(el.first);
    output_tensor_names_.push_back(ti.name());
    output_types_.push_back(DataType_Name(ti.dtype()));
    TF_Output oper_out = ParseTensorName(ti.name());
    outputs_.push_back(oper_out);
    // fill output_tensors_ vector with nulls
    output_tensors_.push_back(nullptr);
  }
  num_inputs_ = input_names_.size();
  num_outputs_ = output_names_.size();
  input_shapes_.resize(num_inputs_);
  std::string msg = "Found " + std::to_string(num_inputs_) + " inputs: ";
  for (int i = 0; i < num_inputs_; i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += input_names_[i];
  }
  LOG(INFO) << msg;
  msg = "Found " + std::to_string(num_outputs_) + " outputs: ";
  for (int i = 0; i < num_outputs_; i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += output_names_[i];
  }
  LOG(INFO) << msg;
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
  TF_Buffer* meta_graph = TF_NewBuffer();
  const char* tags = "serve";
  int ntags = 1;
  sess_ = TF_LoadSessionFromSavedModel(sess_opts, run_opts, model_path.c_str(), &tags, ntags,
                                       graph_, meta_graph, status_);
  if (TF_GetCode(status_) != TF_OK) {
    LOG(FATAL) << "ERROR: Unable to create Session " << TF_Message(status_);
    return;  // unreachable
  }
  tensorflow::MetaGraphDef metagraph_def;
  metagraph_def.ParseFromArray(meta_graph->data, meta_graph->length);
  TF_DeleteBuffer(meta_graph);
  const tensorflow::SignatureDef& serving_default_def =
      metagraph_def.signature_def().at("serving_default");

  DetectInputsAndOutputs(serving_default_def.inputs(), serving_default_def.outputs());

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

int Tensorflow2Model::GetOutputIndex(const char* name) const {
  for (int i = 0; i < output_names_.size(); i++) {
    if (output_names_[i].compare(name) == 0) {
      return i;
    }
  }
  LOG(FATAL) << "Output Tensor not found, name: " << name;
  return -1;  // unreachable
}

void Tensorflow2Model::GetOutputByName(const char* name, void* out) {
  int index = GetOutputIndex(name);
  GetOutput(index, out);
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
