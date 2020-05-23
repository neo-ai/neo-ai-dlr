#include "dlr_tensorflow/dlr_tensorflow.h"

#include <cstring>
#include <fstream>
#include <numeric>
#include <regex>

using namespace dlr;

std::string dlr::GetTensorflowFile(const std::string& dirname) {
  // Support the case where user provides full path to .pb file.
  if (EndsWith(dirname, ".pb")) {
    return dirname;
  }
  // Scan Dir to find .pb file and check that only one .pb file is provided.
  std::string pb_file;
  std::vector<std::string> paths_vec;
  ListDir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    std::string basename = GetBasename(filename);
    if (EndsWith(filename, ".pb")) {
      if (pb_file.empty()) {
        pb_file = filename;
      } else {
        LOG(FATAL) << "Multiple .pb files under the folder: " << dirname;
      }
    }
  }
  if (pb_file.empty()) {
    LOG(FATAL) << "No Tensorflow frozen model file found under folder: "
               << dirname;
  }
  return pb_file;
}

void dlr::FreeBuffer(void* data, size_t length) { free(data); }

TF_Buffer* dlr::ReadTFFile(const char* file) {
  FILE* f = fopen(file, "rb");
  fseek(f, 0L, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0L, SEEK_SET);  // same as rewind(f);

  void* data = malloc(fsize);
  if (fread(data, fsize, 1, f) != 1) {
    LOG(FATAL) << "ERROR: Unable to read file " << file;
  }
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = FreeBuffer;
  return buf;
}

void dlr::PrepareTFConfigProto(const DLR_TFConfig& tf_config,
                               std::vector<std::uint8_t>& config) {
  if (tf_config.intra_op_parallelism_threads > 0 &&
      tf_config.intra_op_parallelism_threads < 256) {
    // More info https://github.com/tensorflow/tensorflow/issues/13853
    config.insert(config.end(),
                  {0x10, (std::uint8_t)tf_config.intra_op_parallelism_threads});
    LOG(INFO) << "Set intra_op_parallelism_threads to "
              << tf_config.intra_op_parallelism_threads;
  }
  if (tf_config.inter_op_parallelism_threads > 0 &&
      tf_config.inter_op_parallelism_threads < 256) {
    // More info https://github.com/tensorflow/tensorflow/issues/13853
    config.insert(config.end(),
                  {0x28, (std::uint8_t)tf_config.inter_op_parallelism_threads});
    LOG(INFO) << "Set inter_op_parallelism_threads to "
              << tf_config.inter_op_parallelism_threads;
  }

  // Tensorflow GPUOptions
  std::vector<std::uint8_t> gpu_options = {0x32, 0x0};

  double gpu_memory_fraction =
      tf_config.gpu_options.per_process_gpu_memory_fraction;
  if (gpu_memory_fraction > 0.001 && gpu_memory_fraction < 1.0) {
    std::uint8_t proto[9] = {0x9,  0xFF, 0xFF, 0xFF, 0xFF,
                             0xFF, 0xFF, 0xFF, 0xFF};
    auto bytes = reinterpret_cast<std::uint8_t*>(&gpu_memory_fraction);
    // Put it to the config byte-array, from 1 to 9:
    for (std::size_t i = 0; i < sizeof(gpu_memory_fraction); i++) {
      proto[i + 1] = bytes[i];
    }
    gpu_options.insert(gpu_options.end(), proto, proto + 9);
    LOG(INFO) << "Set gpu_options.per_process_gpu_memory_fraction to "
              << gpu_memory_fraction;
  }

  if (tf_config.gpu_options.allow_growth != 0) {
    gpu_options.insert(gpu_options.end(), {0x20, 0x1});
    LOG(INFO) << "Set gpu_options.allow_growth to True";
  }
  // update gpu_options data length (stored in byte #2)
  gpu_options[1] = gpu_options.size() - 2;

  if (gpu_options[1] > 0) {
    config.insert(config.end(), gpu_options.begin(), gpu_options.end());
  }
}

void TensorflowModel::LoadFrozenModel(const char* pb_file) {
  const char* op_prefix = "";
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(opts, op_prefix);

  TF_Buffer* graph_def = ReadTFFile(pb_file);
  TF_GraphImportGraphDef(graph_, graph_def, opts, status_);
  if (TF_GetCode(status_) != TF_OK) {
    LOG(FATAL) << "ERROR: Unable to import graph " << TF_Message(status_);
    return;  // unreachable
  }
  LOG(INFO) << "Successfully imported graph";
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_def);
}

TF_Output TensorflowModel::ParseTensorName(const std::string& t_name) {
  std::regex r("^(.+):(\\d+)$");
  std::smatch match;
  if (!std::regex_search(t_name, match, r)) {
    LOG(FATAL) << "ERROR: failed to parse tensor name " << t_name;
  }
  std::string op_name = match.str(1);
  int op_out_id = std::stoi(match.str(2));
  TF_Operation* op = TF_GraphOperationByName(graph_, op_name.c_str());
  if (op == NULL) {
    LOG(FATAL) << "ERROR: TF_GraphOperationByName failed for operation "
               << op_name;
  }
  TF_Output oper_out = {op, op_out_id};
  return oper_out;
}

void TensorflowModel::DetectInputs() {
  size_t pos = 0;
  TF_Operation* op;
  while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
    const std::string op_type = TF_OperationOpType(op);
    const int n_in = TF_OperationNumInputs(op);
    const int n_out = TF_OperationNumOutputs(op);
    const std::string op_name = TF_OperationName(op);
    if (op_type == "Placeholder" && n_in == 0 && n_out == 1) {
      input_names_.push_back(op_name + ":0");
    }
  }
  num_inputs_ = input_names_.size();
  std::string msg =
      "Found " + std::to_string(num_inputs_) + " possible inputs: ";
  for (int i = 0; i < num_inputs_; i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += input_names_[i];
  }
  LOG(INFO) << msg;
}

void TensorflowModel::DetectOutputs() {
  size_t pos = 0;
  TF_Operation* op;
  // while loop
  while ((op = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
    const std::string op_type = TF_OperationOpType(op);
    const int n_out = TF_OperationNumOutputs(op);
    const int n_cout = TF_OperationNumControlOutputs(op);
    const std::string op_name = TF_OperationName(op);
    if (op_type != "Const" && op_type != "Assign" && op_type != "NoOp" &&
        op_type != "Placeholder" && n_cout == 0) {
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
  std::string msg =
      "Found " + std::to_string(num_outputs_) + " possible outputs: ";
  for (int i = 0; i < num_outputs_; i++) {
    if (i > 0) {
      msg += ", ";
    }
    msg += output_names_[i];
  }
  LOG(INFO) << msg;
}

void TensorflowModel::DetectInputShapes() {
  for (int i = 0; i < num_inputs_; i++) {
    const std::string& t_name = input_names_[i];
    const TF_Output oper_out = ParseTensorName(t_name);

    int n_dim = TF_GraphGetTensorNumDims(graph_, oper_out, status_);
    if (TF_GetCode(status_) != TF_OK) {
      LOG(FATAL) << "ERROR: TF_GraphGetTensorNumDims failed "
                 << TF_Message(status_);
      return;  // unreachable
    }
    int64_t dims[n_dim];
    TF_GraphGetTensorShape(graph_, oper_out, dims, n_dim, status_);
    if (TF_GetCode(status_) != TF_OK) {
      LOG(FATAL) << "ERROR: TF_GraphGetTensorShape failed "
                 << TF_Message(status_);
      return;  // unreachable
    }
    // Set Batch size to 1 if undefined
    if (dims[0] == -1) {
      dims[0] = 1;
    }
    for (int z = 1; z < n_dim; z++) {
      if (dims[z] < 1) {
        LOG(FATAL) << "ERROR: non-positive dimensions are not supported. "
                   << "tensor: " << t_name << ", dims[" << z << "]=" << dims[z];
        return;  // unreachable
      }
    }
    input_shapes_.push_back(std::vector<int64_t>(dims, dims + n_dim));
  }
}

void TensorflowModel::PrepInputs() {
  for (int i = 0; i < num_inputs_; i++) {
    const std::string& t_name = input_names_[i];
    const TF_Output oper_out = ParseTensorName(t_name);
    const std::vector<int64_t>& shape = input_shapes_[i];
    const int64_t* dims = shape.data();
    const int n_dim = shape.size();
    size_t num_elements = 1;
    for (int z = 0; z < n_dim; z++) {
      if (dims[z] < 1) {
        LOG(FATAL) << "ERROR: non-positive dimensions are not supported. "
                   << "tensor: " << t_name << ", dims[" << z << "]=" << dims[z];
        return;  // unreachable
      }
      num_elements *= dims[z];
    }
    const TF_DataType t_type = TF_OperationOutputType(oper_out);
    const size_t num_bytes = TF_DataTypeSize(t_type) * num_elements;

    TF_GraphSetTensorShape(graph_, oper_out, dims, n_dim, status_);
    if (TF_GetCode(status_) != TF_OK) {
      LOG(FATAL) << "ERROR: TF_GraphSetTensorShape failed "
                 << TF_Message(status_);
      return;  // unreachable
    }

    TF_Tensor* tensor = TF_AllocateTensor(t_type, dims, n_dim, num_bytes);
    inputs_.push_back(oper_out);
    input_tensors_.push_back(tensor);
  }
  LOG(INFO) << "Input Tensors were allocated";
}

void TensorflowModel::PrepOutputs() {
  for (std::string& t_name : output_names_) {
    TF_Output oper_out = ParseTensorName(t_name);

    outputs_.push_back(oper_out);
    // fill output_tensors_ vector with nulls
    output_tensors_.push_back(nullptr);
  }
}

int TensorflowModel::GetInputId(const char* name) {
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
TensorflowModel::TensorflowModel(
    const std::string& model_path, const DLContext& ctx,
    const std::vector<std::string>& inputs,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::string>& outputs, const DLR_TFConfig& tf_config)
    : DLRModel(ctx, DLRBackend::kTENSORFLOW) {
  const std::string pb_file = GetTensorflowFile(model_path);

  status_ = TF_NewStatus();
  graph_ = TF_NewGraph();

  LoadFrozenModel(pb_file.c_str());

  if (inputs.empty()) {
    DetectInputs();
  } else {
    input_names_ = inputs;
    num_inputs_ = input_names_.size();
  }
  if (input_shapes.empty()) {
    DetectInputShapes();
  } else {
    input_shapes_ = input_shapes;
  }
  if (outputs.empty()) {
    DetectOutputs();
  } else {
    output_names_ = outputs;
    num_outputs_ = output_names_.size();
  }

  PrepInputs();
  PrepOutputs();

  LOG(INFO) << "Tensorflow Model was created";

  TF_SessionOptions* opts = TF_NewSessionOptions();
  std::vector<std::uint8_t> config;
  PrepareTFConfigProto(tf_config, config);
  if (!config.empty()) {
    TF_SetConfig(opts, config.data(), config.size(), status_);
    if (TF_GetCode(status_) != TF_OK) {
      TF_DeleteSessionOptions(opts);
      LOG(FATAL) << "ERROR: TF_SetConfig failed " << TF_Message(status_);
      return;  // unreachable
    }
  }

  sess_ = TF_NewSession(graph_, opts, status_);
  if (TF_GetCode(status_) != TF_OK) {
    LOG(FATAL) << "ERROR: Unable to create Session " << TF_Message(status_);
    return;  // unreachable
  }
  TF_DeleteSessionOptions(opts);
  LOG(INFO) << "Tensorflow Session was created";
}

// Destructor
TensorflowModel::~TensorflowModel() {
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

  LOG(INFO) << "TensorflowModel was deleted";
}

std::vector<std::string> TensorflowModel::GetWeightNames() const {
  LOG(FATAL) << "GetWeightNames is not supported by Tensorflow backend";
  return std::vector<std::string>();  // unreachable
}

const char* TensorflowModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* TensorflowModel::GetInputType(int index) const {
  LOG(FATAL) << "GetInputType is not supported by Tensorflow backend";
  return "";  // unreachable
}

const char* TensorflowModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Tensorflow backend";
  return "";  // unreachable
}

void TensorflowModel::SetInput(const char* name, const int64_t* shape,
                               void* input, int dim) {
  int index = GetInputId(name);
  TF_Tensor* tensor = input_tensors_[index];
  CHECK_EQ(dim, TF_NumDims(tensor)) << "Incorrect input dim";
  for (int i = 0; i < dim; i++) {
    CHECK_EQ(shape[i], TF_Dim(tensor, i)) << "Incorrect input shape";
  }
  size_t num_bytes = TF_TensorByteSize(tensor);
  void* in_t_data = TF_TensorData(tensor);
  std::memcpy(in_t_data, input, num_bytes);
}

void TensorflowModel::GetInput(const char* name, void* input) {
  int index = GetInputId(name);
  TF_Tensor* tensor = input_tensors_[index];
  size_t num_bytes = TF_TensorByteSize(tensor);
  void* in_t_data = TF_TensorData(tensor);
  std::memcpy(input, in_t_data, num_bytes);
}

void TensorflowModel::GetOutputShape(int index, int64_t* shape) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  int n_dim = TF_NumDims(tensor);
  for (int i = 0; i < n_dim; i++) {
    shape[i] = TF_Dim(tensor, i);
  }
}

void TensorflowModel::GetOutput(int index, void* output) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  size_t num_bytes = TF_TensorByteSize(tensor);
  void* out_t_data = TF_TensorData(tensor);
  std::memcpy(output, out_t_data, num_bytes);
}

void TensorflowModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  *size = TF_TensorElementCount(tensor);
  *dim = TF_NumDims(tensor);
}

const char* TensorflowModel::GetOutputType(int index) const {
  LOG(FATAL) << "GetOutputType is not supported by Tensorflow backend";
  return "";  // unreachable
}

void TensorflowModel::Run() {
  // Delete previous output Tensors to prevent GPU memory leak
  for (TF_Tensor* tensor : output_tensors_) {
    if (tensor != nullptr) {
      TF_DeleteTensor(tensor);
    }
  }
  TF_SessionRun(sess_,
                NULL,  // Run options.
                inputs_.data(), input_tensors_.data(), num_inputs_,
                outputs_.data(), output_tensors_.data(), num_outputs_, NULL,
                0,       // Target operations, number of targets.
                NULL,    // Run metadata.
                status_  // Output status.
  );
  if (TF_GetCode(status_) != TF_OK) {
    LOG(FATAL) << "ERROR: Session run error: " << TF_Message(status_);
    return;  // unreachable
  }
}

const char* TensorflowModel::GetBackend() const { return "tensorflow"; }

void TensorflowModel::SetNumThreads(int threads) {
  LOG(FATAL) << "SetNumThreads is not supported by Tensorflow backend";
}

void TensorflowModel::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by Tensorflow backend";
}
