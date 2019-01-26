#include <algorithm>
#include <cmath>
#include <dlr.h>
#include <fstream>
#include <streambuf>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <numeric>

namespace {

/* The folloing file names are reserved by SageMaker and should not be used
 * as model JSON */
constexpr const char* SAGEMAKER_AUXILIARY_JSON_FILES[] = {
  "model-shapes.json", "hyperparams.json"
};

}

using namespace dlr;

void listdir(const std::string& dirname, std::vector<std::string> &paths) {
  DIR* dirp = opendir(dirname.c_str());
  if(!dirp) {
    LOG(FATAL) << "Invalid folder path: " << dirname;
  }
  struct dirent * dp;
  while ((dp = readdir(dirp)) != NULL) {
    paths.push_back(dp->d_name);
  }
  closedir(dirp);
}

bool IsFileEmpty(const std::string &filePath){
  std::ifstream pFile(filePath);
  return pFile.peek() == std::ifstream::traits_type::eof();
}

std::string get_version(const std::string &json_path) {
  std::ifstream file(json_path);
  bool colon_flag = false;
  bool quote_flag = false; std::string version = "";
  if (file.is_open()) {
    char c;
    while (file.good()) {
      c = file.get();
      if (c == ':')
        colon_flag = true;
      else if (colon_flag && quote_flag && (c == '"'))
        return version;
      else if (colon_flag && quote_flag)
        version.push_back(c);
    }
  }
  return version;
}

ModelPath get_treelite_paths(const std::string &dirname) {
  ModelPath paths;
  std::vector<std::string> paths_vec;
  listdir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    if (endsWith(filename, ".so")) {
      paths.model_lib = dirname + "/" + filename;
    } else if (filename == "version.json") {
      paths.ver_json = dirname + "/" + filename;
    }
  }
  if ( paths.model_lib.empty() ){
    LOG(FATAL) << "No valid Treelite model files found under folder:" << dirname;
  }
  return paths;
}

ModelPath get_tvm_paths(const std::string &dirname) {
  ModelPath paths;
  std::vector<std::string> paths_vec;
  listdir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    if (endsWith(filename, ".json")
        && std::all_of(std::begin(SAGEMAKER_AUXILIARY_JSON_FILES),
                       std::end(SAGEMAKER_AUXILIARY_JSON_FILES),
                       [filename](const std::string& s)
                                 { return (s != filename); })
        && filename != "version.json") {
      paths.model_json = dirname + "/" + filename;
    } else if (endsWith(filename, ".so")) {
      paths.model_lib = dirname + "/" + filename;
    } else if (endsWith(filename, ".params")) {
      paths.params = dirname + "/" +filename;
    } else if (filename == "version.json") {
      paths.ver_json = dirname + "/" +filename;
    }
  }
  if ( paths.model_json.empty() || paths.model_lib.empty() || paths.params.empty() ){
    LOG(FATAL) << "No valid TVM model files found under folder:" << dirname;
  }
  return paths;
}

DLRBackend get_backend(const std::string &dirname) {
  std::vector<std::string> paths;
  listdir(dirname, paths);
  DLRBackend backend = DLRBackend::kTREELITE;
  for (auto filename: paths) {
    if (endsWith(filename, ".params")){
      backend = DLRBackend::kTVM;
      break;
    }
  }
  return backend;
}

DLRModel::DLRModel(const std::string& model_path,
                   const DLContext& ctx) {
  backend_ = get_backend(model_path);
  ctx_ = ctx;

//  std::string cmdline = "tar -xf " + model_path; 
//  system(cmdline.c_str());
  if (backend_ == DLRBackend::kTVM) {
    SetupTVMModule(model_path);
  } else if (backend_ == DLRBackend::kTREELITE) {
    SetupTreeliteModule(model_path);
  } else {
    LOG(FATAL) << "Unsupported backend!";
  }
}

void DLRModel::SetupTVMModule(const std::string& model_path) {
  ModelPath paths = get_tvm_paths(model_path);
  std::ifstream jstream(paths.model_json);
  std::stringstream json_blob;
  json_blob << jstream.rdbuf();
  std::ifstream pstream(paths.params);
  std::stringstream param_blob;
  param_blob << pstream.rdbuf();

  tvm::runtime::Module module;
  if (!IsFileEmpty(paths.model_lib)){
    module = tvm::runtime::Module::LoadFromFile(paths.model_lib);
  }
  tvm_graph_runtime_ =
    std::make_shared<tvm::runtime::GraphRuntime>();
  tvm_graph_runtime_->Init(json_blob.str(), module, {ctx_});
  tvm_graph_runtime_->LoadParams(param_blob.str());

  tvm_module_ = std::make_shared<tvm::runtime::Module>(
      tvm::runtime::Module(tvm_graph_runtime_));

  // Save the number of inputs. It excludes inputs that could be obtained
  // through the param file, such as weights.
  num_inputs_ = tvm_graph_runtime_->NumInputs() - GetWeightNames().size();
  std::vector<std::string> input_names;
  for (int i = 0; i < num_inputs_; i++)  {
    input_names.push_back(tvm_graph_runtime_->GetInputName(i));
  }
  std::vector<std::string> weight_names = tvm_graph_runtime_->GetWeightNames();
  std::set_difference(input_names.begin(), input_names.end(),
                      weight_names.begin(), weight_names.end(),
                      std::inserter(input_names_, input_names_.begin()));

  // Get the number of output and reserve space to save output tensor
  // pointers.
  num_outputs_ = tvm_graph_runtime_->NumOutputs();
    outputs_.resize(num_outputs_);
  for (int i = 0; i < num_outputs_; i++) {
    tvm::runtime::NDArray output = tvm_graph_runtime_->GetOutput(i);
    outputs_[i] = output.operator->();
  }
}

void DLRModel::SetupTreeliteModule(const std::string& model_path) {
  ModelPath paths = get_treelite_paths(model_path);
  int num_worker_threads = -1; // use the maximum amount of threads
  int include_master_thread = 1;
  num_inputs_ = 1;
  num_outputs_ = 1;
  // Give a dummy input name to Treelite model.
  input_names_.push_back("data");
  CHECK_EQ(TreelitePredictorLoad(paths.model_lib.c_str(),
                                 num_worker_threads,
                                 include_master_thread,
                                 &treelite_model_), 0) << TreeliteGetLastError();
  // NOTE: assume batch size is 1. However, Treelite internally can support
  //       arbitrary batch size
  CHECK_EQ(
      TreelitePredictorQueryNumFeature(treelite_model_, &treelite_num_feature_),
      0)
      << TreeliteGetLastError();
  treelite_input_ = std::unique_ptr<TreelitePredictorEntry[]>(
      new TreelitePredictorEntry[1 * treelite_num_feature_]);

  size_t num_output_class;  // > 1 for multi-class classification; 1 otherwise
  CHECK_EQ(TreelitePredictorQueryNumOutputGroup(treelite_model_,
                                                &num_output_class),
           0)
      << TreeliteGetLastError();
  treelite_output_ =
      std::unique_ptr<float[]>(new float[1 * num_output_class]);
  // NOTE: second dimension of the output shape is smaller than num_output_class
  //       when a multi-class classifier outputs only the class prediction (argmax)
  //       To detect this edge case, run TreelitePredictorPredictInst() once.
  CHECK_EQ(TreelitePredictorPredictInst(treelite_model_, treelite_input_.get(),
                                        0, treelite_output_.get(),
                                        &treelite_output_size_), 0)
    << TreeliteGetLastError();
  CHECK_LE(treelite_output_size_, num_output_class) << "Precondition violated";
  // version_ = get_version(paths.ver_json);
}

std::vector<std::string> DLRModel::GetWeightNames() const {
  if (backend_ != DLRBackend::kTVM) {
    LOG(FATAL) << "Only TVM models have weight file";
  }
  return tvm_graph_runtime_->GetWeightNames();
}

void DLRModel::GetNumInputs(int* num_inputs) const {
  *num_inputs = num_inputs_;
}

const char* DLRModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  if (backend_ == DLRBackend::kTVM) {
    return input_names_[index].c_str();
  } else if (backend_ == DLRBackend::kTREELITE) {
    return "data";
  } else {
    LOG(FATAL) << "Unsupported backend!";
    return ""; // unreachable
  }
}

#define CHECK_SHAPE(msg, value, expected) \
  CHECK_EQ(value, expected) << (msg) << ". Value read: " << (value) << ", Expected: " << (expected);

void DLRModel::SetInput(const char* name, const int64_t* shape, float* input,
                        int dim) {
  if (backend_ == DLRBackend::kTVM) {
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
    CHECK_SHAPE("Mismatch found in input data size", read_size,
                expected_size);
    tvm::runtime::PackedFunc set_input = tvm_module_->GetFunction("set_input");
    set_input(str, &input_tensor);
  } else if (backend_ == DLRBackend::kTREELITE) {
    // NOTE: Assume batch size is 1. However, Treelite internally can support
    //       arbitrary batch size
    // NOTE: Assume that missing values are represented by NAN
    CHECK_SHAPE("Mismatch found in input dimension", dim, 2);
    CHECK_SHAPE("Mismatch found in input shape at dimension 0", shape[0], 1);
    // NOTE: If number of columns is less than num_feature, missing columns
    //       will be automatically padded with missing values
    CHECK_LE(static_cast<size_t>(shape[1]), treelite_num_feature_)
      << "Mismatch found in input shape at dimension 1. Value read: "
      << shape[1] << ", Expected: " << treelite_num_feature_ << " or less";
    for (size_t i = 0; i < static_cast<size_t>(shape[1]); ++i) {
      if (std::isnan(input[i])) {
        treelite_input_[i].missing = -1;
      } else {
        treelite_input_[i].fvalue = input[i];
      }
    }
    for (size_t i = static_cast<size_t>(shape[1]);
                i < treelite_num_feature_; ++i) {
      treelite_input_[i].missing = -1;
    }
  } else {
    LOG(FATAL) << "Unsupported backend!";
  }
}

void DLRModel::GetOutputShape(int index, int64_t* shape) const {
  if (backend_ == DLRBackend::kTVM) {
    std::memcpy(shape, outputs_[index]->shape,
                sizeof(int64_t) * outputs_[index]->ndim);
  } else if (backend_ == DLRBackend::kTREELITE) {
    // NOTE: Assume batch size is 1. However, Treelite internally can support
    //       arbitrary batch size
    shape[0] = 1;
    shape[1] = static_cast<int64_t>(treelite_output_size_);
  } else {
    LOG(FATAL) << "Unsupported backend!";
  }
}

void DLRModel::GetOutput(int index, float* out) {
  if (backend_ == DLRBackend::kTVM) {
    DLTensor output_tensor = *outputs_[index];
    output_tensor.ctx = DLContext{kDLCPU, 0};
    output_tensor.data = out;
    tvm::runtime::PackedFunc get_output = tvm_module_->GetFunction("get_output");
    get_output(index, &output_tensor);
  } else if (backend_ == DLRBackend::kTREELITE) {
    // NOTE: Assume batch size is 1. However, Treelite internally can support
    //       arbitrary batch size
    std::memcpy(out, treelite_output_.get(),
                sizeof(float) * 1 * treelite_output_size_);
  } else {
    LOG(FATAL) << "Unsupported backend!";
  }
}

void DLRModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  if (backend_ == DLRBackend::kTVM) {
    *size = 1;
    const DLTensor* tensor = outputs_[index];
    for (int i = 0; i < tensor->ndim; ++i) {
      *size *= tensor->shape[i];
    }
    *dim = tensor->ndim;
  } else if (backend_ == DLRBackend::kTREELITE) {
    // NOTE: Assume batch size is 1. However, Treelite internally can support
    //       arbitrary batch size
    *size = 1 * static_cast<int64_t>(treelite_output_size_);
    *dim = 2;
  } else {
    LOG(FATAL) << "Unsupported backend!";
  }
}

void DLRModel::GetNumOutputs(int* num_outputs) const {
  *num_outputs = num_outputs_;
}

void DLRModel::Run() {
  if (backend_ == DLRBackend::kTVM) {
    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = tvm_module_->GetFunction("run");
    run();
  } else if (backend_ == DLRBackend::kTREELITE) {
    // NOTE: Assume batch size is 1. However, Treelite internally can support
    //       arbitrary batch size
    size_t out_result_size;
    CHECK_EQ(TreelitePredictorPredictInst(treelite_model_, treelite_input_.get(),
                                          0, treelite_output_.get(),
                                          &out_result_size), 0)
      << TreeliteGetLastError();
  }
}

extern "C" int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetNumInputs(num_inputs);
  API_END();
}

extern "C" int GetDLRInputName(DLRModelHandle* handle, int index,
                               const char** input_name) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  *input_name = model->GetInputName(index);
  API_END();
}

extern "C" int SetDLRInput(DLRModelHandle* handle,
                           const char* name,
                           const int64_t* shape,
                           float* input,
                           int dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->SetInput(name, shape, input, dim);
  API_END();
}

extern "C" int GetDLROutputShape(DLRModelHandle* handle,
                                 int index,
                                 int64_t* shape) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputShape(index, shape);
  API_END();
}

extern "C" int GetDLROutput(DLRModelHandle* handle,
                            int index,
                            float* out) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutput(index, out);
  API_END();
}

extern "C" int GetDLROutputSizeDim(DLRModelHandle* handle,
                                   int index,
                                   int64_t* size,
                                   int* dim) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetOutputSizeDim(index, size, dim);
  API_END();
}

extern "C" int GetDLRNumOutputs(DLRModelHandle* handle,
                                int* num_outputs) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  CHECK(model != nullptr) << "model is nullptr, create it first";
  model->GetNumOutputs(num_outputs);
  API_END();
}

/*! /brief Translate c args from ctypes to std types for DLRModel ctor.
 */
extern "C" int CreateDLRModel(DLRModelHandle* handle,
                              const char* model_path,
                              int dev_type, int dev_id) {
  API_BEGIN();
  const std::string model_path_string(model_path);   
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = dev_id;
  DLRModel *model = new DLRModel(model_path_string, 
                                ctx);
  *handle = model;
  API_END();
}

extern "C" int DeleteDLRModel(DLRModelHandle* handle) {
  API_BEGIN();
  DLRModel* model = static_cast<DLRModel *>(*handle);
  delete model;
  API_END();
}

extern "C" int RunDLRModel(DLRModelHandle *handle) {
  API_BEGIN();
  static_cast<DLRModel *>(*handle)->Run();
  API_END();
}

extern "C" const char* DLRGetLastError() {
  return TVMGetLastError();
}

