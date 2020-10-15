#include "dlr_hexagon/dlr_hexagon.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>

using namespace dlr;

std::string dlr::GetHexagonModelFile(const std::string& dirname) {
  // Support the case where user provides full path to _hexagon_model.so file.
  if (EndsWith(dirname, "_hexagon_model.so")) {
    return dirname;
  }
  // Scan Dir to find _hexagon_model.so file
  std::string hexagon_model_so_file;
  std::vector<std::string> paths_vec;
  ListDir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    std::string basename = GetBasename(filename);
    if (EndsWith(basename, "_hexagon_model.so")) {
      if (hexagon_model_so_file.empty()) {
        hexagon_model_so_file = filename;
      } else {
        LOG(FATAL) << "Multiple _hexagon_model.so files under the folder: "
                   << dirname;
      }
    }
  }
  if (hexagon_model_so_file.empty()) {
    LOG(FATAL) << "No _hexagon_model.so file found under folder: " << dirname;
  }
  return hexagon_model_so_file;
}

bool dlr::FindHexagonNNSkelFile(const std::string& dirname) {
  // Scan Dir to find libhexagon_nn_skel.so
  std::vector<std::string> paths_vec;
  ListDir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    std::string basename = GetBasename(filename);
    if (basename == "libhexagon_nn_skel.so") {
      return true;
    }
  }
  LOG(INFO) << "libhexagon_nn_skel.so file is not found under folder: "
            << dirname;
  return false;
}

void* dlr::FindSymbol(void* handle, const char* fn_name) {
  LOG(INFO) << "Loading " << fn_name;
  void* fn = dlsym(handle, fn_name);
  if (!fn) {
    LOG(FATAL) << "dlsym error for " << fn_name << ":" << dlerror();
  }
  return fn;
}

void HexagonModel::PrintHexagonNNLog() {
  int err = (*dlr_hexagon_nn_getlog)(graph_id_, (unsigned char*)log_buf_,
                                     log_buf_size_);
  if (err == 0) {
    LOG(INFO) << log_buf_;
  }
}

void HexagonModel::GenTensorSpec(bool isInput) {
  int err = 0;
  int id = 0;
  char* name = NULL;
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
  return -1;  // unreachable
}

// Constructor
HexagonModel::HexagonModel(const std::string& model_path, const DLContext& ctx,
                           const int debug_level)
    : DLRModel(ctx, DLRBackend::kHEXAGON) {
  const std::string model_so_file = GetHexagonModelFile(model_path);
  LOG(INFO) << "Model: " << model_so_file;
  const std::string model_folder = GetParentFolder(model_so_file);
  if (FindHexagonNNSkelFile(model_folder)) {
    char* model_folder_abs = realpath(model_folder.c_str(), NULL);
    LOG(INFO) << "ADSP_LIBRARY_PATH=" << model_folder_abs;
    setenv("ADSP_LIBRARY_PATH", model_folder_abs, 1);
    free(model_folder_abs);
  } else {
    LOG(INFO)
        << "libhexagon_nn_skel.so file is not found. User needs to set "
           "ADSP_LIBRARY_PATH to point to libhexagon_nn_skel.so file folder";
  }

  void* handle = dlopen(model_so_file.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    LOG(FATAL) << "Model file open error: " << dlerror();
    return;  // unreachable
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

  graph_id_ = 0;
  input_ = NULL;
  output_ = NULL;
  log_buf_ = NULL;
  debug_level_ = debug_level;
  int err = 0;
  log_buf_size_ = 2 * 1024 * 1024;
  log_buf_ = new char[log_buf_size_];
  if (log_buf_ == NULL) {
    LOG(FATAL) << "Can not allocate print buffer, size: " << log_buf_size_;
    return;  // unreachable
  }

  err = (*dlr_hexagon_model_init)(&graph_id_, &input_, &output_, debug_level_);
  if (err != 0) {
    PrintHexagonNNLog();
    LOG(FATAL) << "dlr_hexagon_model_init failed: " << err;
    return;  // unreachable
  }
  PrintHexagonNNLog();

  // Save the number of inputs
  GenTensorSpec(true /*isInput*/);
  num_inputs_ = input_tensors_spec_.size();

  // Save the number of outputs
  GenTensorSpec(false /*isInput*/);
  num_outputs_ = output_tensors_spec_.size();
  UpdateInputShapes();
  LOG(INFO) << "HexagonModel was created";
  auto metadata = GetMetadataFile(model_folder);
  if (!metadata.empty() && !IsFileEmpty(metadata)) {
    LOG(INFO) << "Loading metadata file: " << metadata;
    LoadJsonFromFile(metadata, this->metadata_);
    ValidateDeviceTypeIfExists();
  } else {
    LOG(INFO) << "No metadata found";
  }
}

// Destructor
HexagonModel::~HexagonModel() {
  if (graph_id_ != 0 && dlr_hexagon_model_close != NULL) {
    (*dlr_hexagon_model_close)(graph_id_);
    input_ = NULL;
    output_ = NULL;
    graph_id_ = 0;
  }
  if (log_buf_ != NULL) {
    delete[] log_buf_;
    log_buf_ = NULL;
  }
  LOG(INFO) << "HexagonModel was deleted";
}
void HexagonModel::UpdateInputShapes() {
  input_shapes_.resize(num_inputs_);
  for (auto i = 0; i < num_inputs_; i++) {
    std::vector<int64_t> input_shape(input_tensors_spec_[i].shape.begin(),
                                     input_tensors_spec_[i].shape.end());
    input_shapes_[i] = input_shape;
  }
}

std::vector<std::string> HexagonModel::GetWeightNames() const {
  LOG(FATAL) << "GetWeightNames is not supported by Hexagon backend";
  return std::vector<std::string>();  // unreachable
}

const char* HexagonModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index].c_str();
}
const int HexagonModel::GetInputDim(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index].size();
}

const int64_t HexagonModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return std::accumulate(input_shapes_[index].begin(), input_shapes_[index].end(), 1,
                         std::multiplies<int64_t>());
}

const char* HexagonModel::GetInputType(int index) const {
  LOG(FATAL) << "GetInputType is not supported by Hexagon backend";
  return "";  // unreachable
}

const char* HexagonModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Hexagon backend";
  return "";  // unreachable
}

void HexagonModel::SetInput(const char* name, const int64_t* shape,
                            void* input, int dim) {
  int index = GetInputId(name);

  // Check Size and Dim
  CHECK_EQ(dim, input_tensors_spec_[index].dim) << "Incorrect input dim";
  for (int i = 0; i < dim; i++) {
    CHECK_EQ(shape[i], input_tensors_spec_[index].shape[i])
        << "Incorrect input shape";
  }
  std::memcpy(input_, input, input_tensors_spec_[index].bytes);
}

void HexagonModel::GetInput(const char* name, void* input) {
  int index = GetInputId(name);
  std::memcpy(input, input_, input_tensors_spec_[index].bytes);
}

void HexagonModel::GetOutputShape(int index, int64_t* shape) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  for (int i = 0; i < output_tensors_spec_[index].dim; i++) {
    shape[i] = (int64_t)output_tensors_spec_[index].shape[i];
  }
}

void HexagonModel::GetOutput(int index, void* out) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  std::memcpy(out, output_, output_tensors_spec_[index].bytes);
}

const void* HexagonModel::GetOutputPtr(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_;
}

void HexagonModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  *size = output_tensors_spec_[index].size;
  *dim = output_tensors_spec_[index].dim;
}

const char* HexagonModel::GetOutputType(int index) const {
  LOG(FATAL) << "GetOutputType is not supported by Hexagon backend";
  return "";  // unreachable
}

void HexagonModel::Run() {
  int err = (*dlr_hexagon_model_exec)(graph_id_, input_, output_);
  // Invoke
  if (err != 0) {
    LOG(FATAL) << "Failed to exec hexagon model: " << err;
    return;  // unreachable
  }
}

const char* HexagonModel::GetBackend() const { return "hexagon"; }

void HexagonModel::SetNumThreads(int threads) {
  LOG(FATAL) << "SetNumThreads is not supported by Hexagon backend";
}

void HexagonModel::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by Hexagon backend";
}
