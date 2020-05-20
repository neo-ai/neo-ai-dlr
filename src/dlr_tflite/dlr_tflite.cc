#include "dlr_tflite/dlr_tflite.h"

#include <fstream>
#include <numeric>

using namespace dlr;

// TFLite Type names
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h
static std::string TFLITE_TYPES_STR[12]{
    /* 0 */ "none",  /* 1 */ "float32",  /* 2 */ "int32",
    /* 3 */ "uint8", /* 4 */ "int64",    /* 5 */ "string",
    /* 6 */ "bool",  /* 7 */ "int16",    /* 8 */ "complex64",
    /* 9 */ "int8",  /* 10 */ "float16", /* 11 */ "float64"};

std::string dlr::GetTFLiteFile(const std::string& dirname) {
  // Support the case where user provides full path to tflite file.
  if (EndsWith(dirname, ".tflite")) {
    return dirname;
  }
  // Scan Dir to find tflite file and check that only one tflite file is
  // provided.
  std::string tflite_file;
  std::vector<std::string> paths_vec;
  ListDir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    std::string basename = GetBasename(filename);
    if (EndsWith(filename, ".tflite")) {
      if (tflite_file.empty()) {
        tflite_file = filename;
      } else {
        LOG(FATAL) << "Multiple .tflite files under the folder: " << dirname;
      }
    }
  }
  if (tflite_file.empty()) {
    LOG(FATAL) << "No TFLite model file found under folder: " << dirname;
  }
  return tflite_file;
}

void TFLiteModel::GenTensorSpec(bool isInput) {
  std::vector<int> t_ids =
      isInput ? interpreter_->inputs() : interpreter_->outputs();
  size_t n = t_ids.size();
  for (int i = 0; i < n; i++) {
    auto t_id = t_ids[i];
    TfLiteTensor* tensor = interpreter_->tensor(t_id);
    TensorSpec t_spec;
    t_spec.id = t_id;
    t_spec.name = std::string(tensor->name);
    t_spec.type = tensor->type;
    t_spec.dim = tensor->dims->size;
    // Use vector(first, last) to copy int* to vector. Do not keep pointers of
    // internal TF data structures.
    t_spec.shape = std::vector<int>(tensor->dims->data,
                                    tensor->dims->data + tensor->dims->size);
    t_spec.bytes = tensor->bytes;
    int64_t t_sz = 1;
    for (int j = 0; j < t_spec.dim; j++) {
      t_sz *= t_spec.shape[j];
    }
    t_spec.size = t_sz;
    if (isInput) {
      input_tensors_spec_.push_back(t_spec);
      // Fill in input_names_ vector as well because it is defined in base class
      // DLRModel
      input_names_.push_back(t_spec.name);
    } else {
      output_tensors_spec_.push_back(t_spec);
    }
  }
}

int TFLiteModel::GetInputId(const char* name) {
  // In most of the cases it will be just 1 element in the vector.
  // Scan vector to find tensor by name.
  for (int i = 0; i < num_inputs_; i++) {
    if (input_tensors_spec_[i].name.compare(name) == 0) {
      return i;
    }
  }
  LOG(FATAL) << "Input Tensor not found, name: " << name;
  return -1;  // unreachable
}

// Constructor
TFLiteModel::TFLiteModel(const std::string& model_path, const DLContext& ctx,
                         const int threads, const bool use_nnapi)
    : DLRModel(ctx, DLRBackend::kTFLITE) {
  const std::string tflite_file = GetTFLiteFile(model_path);

  // ensure the model and error_reporter lifetime is at least as long as
  // interpreter's lifetime
  error_reporter_ = new tflite::StderrReporter();
  model_ = tflite::FlatBufferModel::BuildFromFile(tflite_file.c_str(),
                                                  error_reporter_);
  if (!model_) {
    LOG(FATAL) << "Failed to load the model: " << tflite_file;
    return;  // unreachable
  }
  // op_resolver does not need to exist for the duration of interpreter objects.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  if (builder(&interpreter_) != kTfLiteOk) {
    LOG(FATAL) << "Failed to build TFLite Interpreter";
    return;  // unreachable
  }
  interpreter_->UseNNAPI(use_nnapi);
  LOG(INFO) << "Use NNAPI: " << (use_nnapi ? "true" : "false");
  if (threads > 0) {
    interpreter_->SetNumThreads(threads);
    LOG(INFO) << "Set Num Threads: " << threads;
  }

  // AllocateTensors
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors";
    return;  // unreachable
  }
  LOG(INFO) << "AllocateTensors - OK";

  // Save the number of inputs
  num_inputs_ = interpreter_->inputs().size();
  GenTensorSpec(true);

  // Save the number of outputs
  num_outputs_ = interpreter_->outputs().size();
  GenTensorSpec(false);

  LOG(INFO) << "TFLiteModel was created";
}

// Destructor
TFLiteModel::~TFLiteModel() {
  interpreter_.reset();
  model_.reset();
  delete error_reporter_;
  LOG(INFO) << "TFLiteModel was deleted";
}

std::vector<std::string> TFLiteModel::GetWeightNames() const {
  LOG(FATAL) << "GetWeightNames is not supported by TFLite backend";
  return std::vector<std::string>();  // unreachable
}

const char* TFLiteModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}

const char* TFLiteModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  TensorSpec input_tensors_spec = input_tensors_spec_[index];
  int type_id = input_tensors_spec.type;
  return TFLITE_TYPES_STR[type_id].c_str();
}

const char* TFLiteModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by TFLite backend";
  return "";  // unreachable
}

void TFLiteModel::SetInput(const char* name, const int64_t* shape, float* input,
                           int dim) {
  int index = GetInputId(name);
  TensorSpec input_tensors_spec = input_tensors_spec_[index];
  // Check Size and Dim
  CHECK_EQ(dim, input_tensors_spec.dim) << "Incorrect input dim";
  for (int i = 0; i < dim; i++) {
    CHECK_EQ(shape[i], input_tensors_spec.shape[i]) << "Incorrect input shape";
  }
  void* in_t_data;
  if (input_tensors_spec.type == TfLiteType::kTfLiteFloat32) {
    in_t_data = interpreter_->typed_input_tensor<float>(index);
  } else if (input_tensors_spec.type == TfLiteType::kTfLiteUInt8) {
    in_t_data = interpreter_->typed_input_tensor<uint8_t>(index);
  } else {
    LOG(FATAL) << "Input tensor type is not supported by TFLite backend";
  }
  std::memcpy(in_t_data, input, input_tensors_spec_[index].bytes);
}

void TFLiteModel::GetInput(const char* name, float* input) {
  int index = GetInputId(name);
  TensorSpec input_tensors_spec = input_tensors_spec_[index];
  void* in_t_data;
  if (input_tensors_spec.type == TfLiteType::kTfLiteFloat32) {
    in_t_data = interpreter_->typed_input_tensor<float>(index);
  } else if (input_tensors_spec.type == TfLiteType::kTfLiteUInt8) {
    in_t_data = interpreter_->typed_input_tensor<uint8_t>(index);
  } else {
    LOG(FATAL) << "Input tensor type is not supported by TFLite backend";
  }
  std::memcpy(input, in_t_data, input_tensors_spec.bytes);
}

void TFLiteModel::GetOutputShape(int index, int64_t* shape) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  for (int i = 0; i < output_tensors_spec_[index].dim; i++) {
    shape[i] = (int64_t)output_tensors_spec_[index].shape[i];
  }
}

void TFLiteModel::GetOutput(int index, float* out) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TensorSpec output_tensors_spec = output_tensors_spec_[index];
  void* out_t_data;
  if (output_tensors_spec.type == TfLiteType::kTfLiteFloat32) {
    out_t_data = interpreter_->typed_output_tensor<float>(index);
  } else if (output_tensors_spec.type == TfLiteType::kTfLiteUInt8) {
    out_t_data = interpreter_->typed_output_tensor<uint8_t>(index);
  } else {
    LOG(FATAL) << "Output tensor type is not supported by TFLite backend";
  }
  std::memcpy(out, out_t_data, output_tensors_spec_[index].bytes);
}

void TFLiteModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  *size = output_tensors_spec_[index].size;
  *dim = output_tensors_spec_[index].dim;
}

const char* TFLiteModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  TensorSpec output_tensors_spec = output_tensors_spec_[index];
  int type_id = output_tensors_spec.type;
  return TFLITE_TYPES_STR[type_id].c_str();
}

void TFLiteModel::Run() {
  // Invoke
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke interpreter!";
    return;  // unreachable
  }
}

const char* TFLiteModel::GetBackend() const { return "tflite"; }

void TFLiteModel::SetNumThreads(int threads) {
  if (threads > 0) {
    interpreter_->SetNumThreads(threads);
    LOG(INFO) << "Set Num Threads: " << threads;
  }
}

void TFLiteModel::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by TFLite backend";
}
