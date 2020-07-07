#include "dlr_hexagon/dlr_hexagon.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>

using namespace dlr;

// START - Helper functions for TVM Model
void HexagonModel::InitModelArtifact(const std::string &path) {
  model_artifact_ = {};

  std::string directory = path;
  bool path_is_directory = true;
  if (EndsWith(path, "_hexagon_model.so")) {
    directory = GetParentFolder(path);
    model_artifact_.model_file = path;
    path_is_directory = false;
  }

  std::vector<std::string>filenames = ListFilesInDirectory(directory);
  for (auto filename: filenames) {
    if(path_is_directory && EndsWith(filename, "_hexagon_model.so")) {
      if (model_artifact_.model_file.empty()) {
        model_artifact_.model_file = filename;
      } else {
        LOG(FATAL) << "Multiple _hexagon_model.so files under the folder: "
                   << path;
      }
    } else if (EndsWith(filename, "libhexagon_nn_skel.so")) {
      model_artifact_.skeleton_file = filename;
    } else if (EndsWith(filename, "compiled.meta")) {
      model_artifact_.metadata = filename;
    }
  }

  if (model_artifact_.model_file.empty()) {
    LOG(FATAL) << "No _hexagon_model.so file found under folder: " << path;
  }

  if (model_artifact_.skeleton_file.empty()) {
    LOG(INFO)
        << "libhexagon_nn_skel.so file is not found. User needs to set "
           "ADSP_LIBRARY_PATH to point to libhexagon_nn_skel.so file folder";
  } else {
    char* model_folder_abs = realpath(path.c_str(), NULL);
    LOG(INFO) << "ADSP_LIBRARY_PATH=" << model_folder_abs;
    setenv("ADSP_LIBRARY_PATH", model_folder_abs, 1);
    free(model_folder_abs);
  }
}

void* dlr::FindSymbol(void* handle, const char* fn_name) {
  LOG(INFO) << "Loading " << fn_name;
  void* fn = dlsym(handle, fn_name);
  if (!fn) {
    LOG(FATAL) << "dlsym error for " << fn_name << ":" << dlerror();
  }
  return fn;
}
// END - Helper functions

HexagonModel::~HexagonModel() {
  if (graph_id_ != 0 && dlr_hexagon_model_close != nullptr) {
    (*dlr_hexagon_model_close)(graph_id_);
    input_ = nullptr;
    output_ = nullptr;
    graph_id_ = 0;
  }
  if (log_buf_ != nullptr) {
    delete[] log_buf_;
    log_buf_ = nullptr;
  }
  LOG(INFO) << "HexagonModel was deleted";
}

void HexagonModel::InitHexagonModel() {
  int err = (*dlr_hexagon_model_init)(&graph_id_, &input_, &output_, debug_level_);
  if (err != 0) {
    PrintHexagonNNLog();
    LOG(FATAL) << "dlr_hexagon_model_init failed: " << err;
    return;  // unreachable
  }
  PrintHexagonNNLog();
}

void HexagonModel::PrintHexagonNNLog() {
  int err = (*dlr_hexagon_nn_getlog)(graph_id_, (unsigned char*)log_buf_,
                                     kLogBufferSize);
  if (err == 0) {
    LOG(INFO) << log_buf_;
  }
}

void HexagonModel::AllocateLogBuffer() {
  log_buf_ = new char[kLogBufferSize];
  if (log_buf_ == nullptr) {
    LOG(FATAL) << "Can not allocate print buffer, size: " << kLogBufferSize;
    return;  // unreachable
  }
}

void HexagonModel::GenTensorSpec(bool isInput) {
  int err = 0;
  int id = 0;
  char* name = nullptr;
  int dim = 0;
  int* shape = NULL;
  int length = 0;
  int bytes = 0;

  while (true) {
    if (isInput) {
      err = (*dlr_hexagon_input_spec)(id, &name, &dim, &shape, &length, &bytes);
    } else {
      err =
          (*dlr_hexagon_output_spec)(id, &name, &dim, &shape, &length, &bytes);
    }
    if (err != 0) break;
    HexagonTensorSpec t_spec;
    t_spec.name = std::string(name);
    t_spec.dim = dim;
    // Use vector(first, last) to copy int* to vector. Do not keep pointers of
    // internal TF data structures.
    t_spec.shape = std::vector<int>(shape, shape + dim);
    t_spec.bytes = bytes;
    t_spec.size = length;
    if (isInput) {
      input_tensors_spec_.push_back(t_spec);
      // Fill in input_names_ vector as well because it is defined in base class
      // DLRModel
      input_names_.push_back(t_spec.name);
    } else {
      output_tensors_spec_.push_back(t_spec);
    }
    id++;
  }
}

int HexagonModel::GetInputId(const char* name) {
  // In most of the cases it will be just 1 element in the vector.
  // Scan vector to find tensor by name.
  for (int i = 0; i < num_inputs_; i++) {
    if (input_tensors_spec_[i].name.compare(name) == 0) {
      return i;
    }
  }
  LOG(FATAL) << "Input Tensor not found, name: " << name;
}

void HexagonModel::LoadSymbols() {
  LOG(INFO) << "Model File " << model_artifact_.model_file;
  void* handle = dlopen(model_artifact_.model_file.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    LOG(FATAL) << "Model file open error: " << dlerror();
  }

  *(void**)(&dlr_hexagon_model_init) =
      FindSymbol(handle, "dlr_hexagon_model_init");
  *(void**)(&dlr_hexagon_model_exec) =
      FindSymbol(handle, "dlr_hexagon_model_exec");
  *(void**)(&dlr_hexagon_model_close) =
      FindSymbol(handle, "dlr_hexagon_model_close");
  *(void**)(&dlr_hexagon_nn_getlog) =
      FindSymbol(handle, "dlr_hexagon_nn_getlog");
  *(void**)(&dlr_hexagon_input_spec) =
      FindSymbol(handle, "dlr_hexagon_input_spec");
  *(void**)(&dlr_hexagon_output_spec) =
      FindSymbol(handle, "dlr_hexagon_output_spec");
}

void HexagonModel::InitInputOutputTensorSpecs() {
  // Save the number of inputs
  GenTensorSpec(true /*isInput*/);
  num_inputs_ = input_tensors_spec_.size();

  // Save the number of outputs
  GenTensorSpec(false /*isInput*/);
  num_outputs_ = output_tensors_spec_.size();

  UpdateInputShapes();
  UpdateOutputShapes();
}

void HexagonModel::UpdateInputShapes() {
  input_shapes_.resize(num_inputs_);
  for (auto i = 0; i < num_inputs_; i++) {
    std::vector<int64_t> input_shape(input_tensors_spec_[i].shape.begin(), input_tensors_spec_[i].shape.end());
    input_shapes_[i] = input_shape;
  }
}

void HexagonModel::UpdateOutputShapes() {
  output_shapes_.resize(num_outputs_);
  for(auto i = 0; i < num_outputs_; i++) {
    std::vector<int64_t> output_shape(output_tensors_spec_[i].shape.begin(), output_tensors_spec_[i].shape.end());
    output_shapes_[i] = output_shape;
  }
}

const std::string& HexagonModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index];
}

const std::string& HexagonModel::GetInputType(int index) const {
  LOG(FATAL) << "GetInputType is not supported by Hexagon backend";
}

const std::string& HexagonModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Hexagon backend";
}


void HexagonModel::SetInput(const int index, const int64_t batch_size, void* input) {
  std::memcpy(input_, input, input_tensors_spec_[index].bytes);

  // Updated input and output shapes to account for batch size.
  UpdateInputShapes();
  UpdateOutputShapes();
}

void HexagonModel::SetInput(std::string name, const int64_t batch_size, void* input) {
  int index = GetInputId(name.c_str());
  SetInput(index, batch_size, input);
}

void HexagonModel::SetInput(const char* name, const int64_t* shape,
                            void* input, int dim) {
  int index = GetInputId(name);
  std::string node_name(name);

  // Check Size and Dim
  CHECK_EQ(dim, GetInputDim(index)) << "Incorrect input dim";
  for (int i = 0; i < dim; i++) {
    CHECK_EQ(shape[i], input_tensors_spec_[index].shape[i])
        << "Incorrect input shape";
  }
  SetInput(name, *shape, input);
}

void HexagonModel::GetInput(const char* name, void* input) {
  int index = GetInputId(name);
  std::memcpy(input, input_, input_tensors_spec_[index].bytes);
}

const int64_t HexagonModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_tensors_spec_[index].size;
}

const int HexagonModel::GetInputDim(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_tensors_spec_[index].dim;
}

const std::vector<int64_t>& HexagonModel::GetInputShape(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index];
}

const std::vector<int64_t>& HexagonModel::GetOutputShape(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_shapes_[index];
}

void HexagonModel::GetOutput(int index, void* out) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  std::memcpy(out, output_, output_tensors_spec_[index].bytes);
}

const int64_t HexagonModel::GetOutputSize(int index) const  {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_tensors_spec_[index].size;
}

const int HexagonModel::GetOutputDim(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_tensors_spec_[index].dim;
}

const std::string& HexagonModel::GetOutputType(int index) const {
  LOG(FATAL) << "GetOutputType is not supported by Hexagon backend";
}

void HexagonModel::Run() {
  int err = (*dlr_hexagon_model_exec)(graph_id_, input_, output_);
  // Invoke
  if (err != 0) {
    LOG(FATAL) << "Failed to exec hexagon model: " << err;
  }
}

void HexagonModel::SetNumThreads(int threads) {
  LOG(FATAL) << "SetNumThreads is not supported by Hexagon backend";
}

void HexagonModel::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by Hexagon backend";
}
