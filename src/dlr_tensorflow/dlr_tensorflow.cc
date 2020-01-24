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
    printf("File read error....\n");
  }
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = FreeBuffer;
  return buf;
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

void TensorflowModel::GenTensorSpec(bool is_input, const int batch_size) {
  std::vector<std::string> tensor_names =
      is_input ? input_names_ : output_names_;
  std::regex r("^(.+):(\\d+)$");
  for (std::string t_name : tensor_names) {
    std::smatch match;
    std::string op_name;
    int op_out_id;
    if (std::regex_search(t_name, match, r)) {
      op_name = match.str(1);
      op_out_id = std::stoi(match.str(2));
    } else {
      LOG(FATAL) << "ERROR: failed to parse tensor name " << t_name;
      return;  // unreachable
    }

    TF_Operation* op = TF_GraphOperationByName(graph_, op_name.c_str());
    if (op == NULL) {
      LOG(FATAL)
          << "ERROR: inputOp TF_GraphOperationByName failed for operation "
          << op_name;
      return;  // unreachable
    }
    TF_Output oper_out = {op, op_out_id};
    const int n_dim = TF_GraphGetTensorNumDims(graph_, oper_out, status_);
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
    // Set fixed batch size if batch size is dynamic
    if (dims[0] == -1) {
      dims[0] = batch_size > 0 ? batch_size : 1;
      TF_GraphSetTensorShape(graph_, oper_out, dims, n_dim, status_);
      if (TF_GetCode(status_) != TF_OK) {
        LOG(FATAL) << "ERROR: TF_GraphSetTensorShape failed "
                   << TF_Message(status_);
        return;  // unreachable
      }
    }
    size_t num_elements = 1;
    for (int z = 0; z < n_dim; z++) {
      if (dims[z] == -1) {
        LOG(FATAL) << "ERROR: dynamic tensor shape is not supported";
        return;  // unreachable
      }
      num_elements *= dims[z];
    }
    TF_DataType t_type = TF_OperationOutputType(oper_out);
    size_t num_bytes = TF_DataTypeSize(t_type) * num_elements;

    TF_Tensor* tensor = TF_AllocateTensor(t_type, dims, n_dim, num_bytes);

    if (is_input) {
      inputs_.push_back(oper_out);
      input_tensors_.push_back(tensor);
    } else {
      outputs_.push_back(oper_out);
      output_tensors_.push_back(tensor);
    }
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
TensorflowModel::TensorflowModel(const std::string& model_path,
                                 const DLContext& ctx,
                                 const std::vector<std::string>& inputs,
                                 const std::vector<std::string>& outputs,
                                 const int batch_size, const int threads)
    : DLRModel(ctx, DLRBackend::kTENSORFLOW) {
  const std::string pb_file = GetTensorflowFile(model_path);

  status_ = TF_NewStatus();
  graph_ = TF_NewGraph();

  LoadFrozenModel(pb_file.c_str());

  // Using assignment operator to copy one vector to other
  input_names_ = inputs;
  output_names_ = outputs;
  num_inputs_ = inputs.size();
  num_outputs_ = outputs.size();

  GenTensorSpec(/*is_input=*/true, batch_size);
  GenTensorSpec(/*is_input=*/false, batch_size);

  LOG(INFO) << "Tensorflow Model was created";

  TF_SessionOptions* opts = TF_NewSessionOptions();
  if (threads > 0 && threads < 256) {
    // More info https://github.com/tensorflow/tensorflow/issues/13853
    std::array<std::uint8_t, 4> config = {
        {0x10, (std::uint8_t)threads, 0x28, (std::uint8_t)threads}};
    TF_SetConfig(opts, config.data(), config.size(), status_);

    if (TF_GetCode(status_) != TF_OK) {
      TF_DeleteSessionOptions(opts);
      LOG(FATAL) << "ERROR: TF_SetConfig failed " << TF_Message(status_);
      return;  // unreachable
    }
    LOG(INFO) << "Set number of threads to " << threads;
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

const char* TensorflowModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Tensorflow backend";
  return "";  // unreachable
}

void TensorflowModel::SetInput(const char* name, const int64_t* shape,
                               float* input, int dim) {
  int index = GetInputId(name);
  TF_Tensor* tensor = input_tensors_[index];
  CHECK_EQ(dim, TF_NumDims(tensor)) << "Incorrect input dim";
  for (int i = 0; i < dim; i++) {
    CHECK_EQ(shape[i], TF_Dim(tensor, i)) << "Incorrect input shape";
  }
  size_t num_bytes = TF_TensorByteSize(tensor);
  float* in_t_data = (float*)TF_TensorData(tensor);
  std::memcpy(in_t_data, input, num_bytes);
}

void TensorflowModel::GetInput(const char* name, float* input) {
  int index = GetInputId(name);
  TF_Tensor* tensor = input_tensors_[index];
  size_t num_bytes = TF_TensorByteSize(tensor);
  float* in_t_data = (float*)TF_TensorData(tensor);
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

void TensorflowModel::GetOutput(int index, float* output) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  size_t num_bytes = TF_TensorByteSize(tensor);
  float* out_t_data = (float*)TF_TensorData(tensor);
  std::memcpy(output, out_t_data, num_bytes);
}

void TensorflowModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TF_Tensor* tensor = output_tensors_[index];
  *size = TF_TensorElementCount(tensor);
  *dim = TF_NumDims(tensor);
}

void TensorflowModel::Run() {
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
