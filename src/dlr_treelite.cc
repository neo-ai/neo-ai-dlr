#include "dlr_treelite.h"

#include <cmath>
#include <cstring>
#include <fstream>

using namespace dlr;

// not used (was used in SetupTreeliteModule)
std::string GetVersion(const std::string& json_path) {
  std::ifstream file(json_path);
  bool colon_flag = false;
  bool quote_flag = false;
  std::string version = "";
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

ModelPath dlr::GetTreelitePaths(std::vector<std::string> dirname) {
  ModelPath paths;
  std::vector<std::string> paths_vec;
  for (auto dir : dirname) {
    ListDir(dir, paths_vec);
  }
  for (auto filename : paths_vec) {
    if (filename != LIBDLR && EndsWith(filename, LIBEXT)) {
      paths.model_lib = filename;
    } else if (filename == "version.json") {
      paths.ver_json = filename;
    }
  }
  if (paths.model_lib.empty()) {
    LOG(INFO) << "No valid Treelite model files found under folder(s):";
    for (auto dir : dirname) {
      LOG(INFO) << dir;
    }
    LOG(FATAL);
  }
  return paths;
}

void TreeliteModel::SetupTreeliteModule(std::vector<std::string> model_path) {
  ModelPath paths = GetTreelitePaths(model_path);
  // If OMP_NUM_THREADS is set, use it to determine number of threads;
  // if not, use the maximum amount of threads
  const char* val = std::getenv("OMP_NUM_THREADS");
  int num_worker_threads = (val ? std::atoi(val) : -1);
  num_inputs_ = 1;
  num_outputs_ = 1;
  // Give a dummy input name to Treelite model.
  input_names_.push_back("data");
  input_types_.push_back("float32");
  CHECK_EQ(TreelitePredictorLoad(paths.model_lib.c_str(), num_worker_threads,
                                 &treelite_model_),
           0)
      << TreeliteGetLastError();
  CHECK_EQ(
      TreelitePredictorQueryNumFeature(treelite_model_, &treelite_num_feature_),
      0)
      << TreeliteGetLastError();
  treelite_input_.reset(nullptr);

  size_t num_output_class;  // > 1 for multi-class classification; 1 otherwise
  CHECK_EQ(
      TreelitePredictorQueryNumOutputGroup(treelite_model_, &num_output_class),
      0)
      << TreeliteGetLastError();
  treelite_output_buffer_size_ = num_output_class;
  treelite_output_.empty();
  // NOTE: second dimension of the output shape is smaller than num_output_class
  //       when a multi-class classifier outputs only the class prediction
  //       (argmax) To detect this edge case, run TreelitePredictorPredictInst()
  //       once.
  std::vector<TreelitePredictorEntry> tmp_in(treelite_num_feature_);
  std::vector<float> tmp_out(num_output_class);
  CHECK_EQ(TreelitePredictorPredictInst(treelite_model_, tmp_in.data(), 0,
                                        tmp_out.data(), &treelite_output_size_),
           0)
      << TreeliteGetLastError();
  CHECK_LE(treelite_output_size_, num_output_class) << "Precondition violated";
  // version_ = GetVersion(paths.ver_json);
}

std::vector<std::string> TreeliteModel::GetWeightNames() const {
  LOG(FATAL) << "GetWeightNames is not supported by Treelite backend";
  return std::vector<std::string>();  // unreachable
}

const char* TreeliteModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return "data";
}

const char* TreeliteModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return "float32";
}

const char* TreeliteModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Treelite backend";
  return "";  // unreachable
}

void TreeliteModel::SetInput(const char* name, const int64_t* shape,
                             void* input, int dim) {
  // NOTE: Assume that missing values are represented by NAN
  CHECK_SHAPE("Mismatch found in input dimension", dim, 2);
  // NOTE: If number of columns is less than num_feature, missing columns
  //       will be automatically padded with missing values
  CHECK_LE(static_cast<size_t>(shape[1]), treelite_num_feature_)
      << "ClientError: Mismatch found in input shape at dimension 1. Value "
         "read: "
      << shape[1] << ", Expected: " << treelite_num_feature_ << " or less";

  const size_t batch_size = static_cast<size_t>(shape[0]);
  const uint32_t num_col = static_cast<uint32_t>(shape[1]);
  treelite_input_.reset(new TreeliteInput);
  CHECK(treelite_input_);
  treelite_input_->row_ptr.push_back(0);
  float* input_f = (float*) input;

  // NOTE: Assume row-major (C) layout
  for (size_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = 0; j < num_col; ++j) {
      if (!std::isnan(input_f[i * num_col + j])) {
        treelite_input_->data.push_back(input_f[i * num_col + j]);
        treelite_input_->col_ind.push_back(j);
      }
    }
    treelite_input_->row_ptr.push_back(treelite_input_->data.size());
  }
  // Post conditions for CSR matrix initialization
  CHECK_EQ(treelite_input_->data.size(), treelite_input_->col_ind.size());
  CHECK_EQ(treelite_input_->data.size(), treelite_input_->row_ptr.back());
  CHECK_EQ(treelite_input_->row_ptr.size(), batch_size + 1);

  // Save dimensions for input
  treelite_input_->num_row = batch_size;
  treelite_input_->num_col = treelite_num_feature_;

  // Register CSR matrix with Treelite backend
  CHECK_EQ(TreeliteAssembleSparseBatch(
               treelite_input_->data.data(), treelite_input_->col_ind.data(),
               treelite_input_->row_ptr.data(), batch_size,
               treelite_num_feature_, &treelite_input_->handle),
           0)
      << TreeliteGetLastError();
}

void TreeliteModel::GetInput(const char* name, void* input) {
  LOG(FATAL) << "GetInput is not supported by Treelite backend";
}

void TreeliteModel::GetOutputShape(int index, int64_t* shape) const {
  // Use -1 if input is yet unspecified and batch size is not known
  shape[0] =
      treelite_input_ ? static_cast<int64_t>(treelite_input_->num_row) : -1;
  shape[1] = static_cast<int64_t>(treelite_output_size_);
}

void TreeliteModel::GetOutput(int index, void* out) {
  CHECK(treelite_input_);
  std::memcpy(
      out, treelite_output_.data(),
      sizeof(float) * (treelite_input_->num_row) * treelite_output_size_);
}

void TreeliteModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  if (treelite_input_) {
    *size =
        static_cast<int64_t>(treelite_input_->num_row * treelite_output_size_);
  } else {
    // Input is yet unspecified and batch is not known
    *size = treelite_output_size_;
  }
  *dim = 2;
}

const char* TreeliteModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return "float32";
}

void TreeliteModel::Run() {
  size_t out_result_size;
  CHECK(treelite_input_);
  treelite_output_.resize(treelite_input_->num_row *
                          treelite_output_buffer_size_);
  CHECK_EQ(TreelitePredictorPredictBatch(
               treelite_model_, treelite_input_->handle, 1, 0, 0,
               treelite_output_.data(), &out_result_size),
           0)
      << TreeliteGetLastError();
}

const char* TreeliteModel::GetBackend() const { return "treelite"; }

void TreeliteModel::SetNumThreads(int threads) {
  LOG(FATAL) << "SetNumThreads is not supported by Treelite backend";
}

void TreeliteModel::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by Treelite backend";
}
