#include "dlr_treelite.h"

#include <cmath>
#include <cstring>
#include <fstream>

using namespace dlr;

const std::string TreeliteModel::INPUT_NAME = "data";
const std::string TreeliteModel::INPUT_TYPE = "float32";
const std::string TreeliteModel::OUTPUT_TYPE = "float32";

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

ModelPath dlr::SetTreelitePaths(const std::vector<std::string>& files) {
  ModelPath paths;
  dlr::InitModelPath(files, &paths);
  if (paths.model_lib.empty()) {
    throw dmlc::Error("Invalid treelite model artifact. Must have .so file.");
  }
  return paths;
}

void TreeliteModel::SetupTreeliteModule(const std::vector<std::string>& model_path) {
  ModelPath paths = SetTreelitePaths(model_path);
  // If OMP_NUM_THREADS is set, use it to determine number of threads;
  // if not, use the maximum amount of threads
  const char* val = std::getenv("OMP_NUM_THREADS");
  int num_worker_threads = (val ? std::atoi(val) : -1);
  num_inputs_ = 1;
  num_outputs_ = 1;
  // Give a dummy input name to Treelite model.
  input_names_.push_back(INPUT_NAME);
  input_types_.push_back(INPUT_TYPE);
  CHECK_EQ(TreelitePredictorLoad(paths.model_lib.c_str(), num_worker_threads, &treelite_model_), 0)
      << TreeliteGetLastError();
  CHECK_EQ(TreelitePredictorQueryNumFeature(treelite_model_, &treelite_num_feature_), 0)
      << TreeliteGetLastError();
  treelite_input_.reset(nullptr);

  const char* output_type;
  CHECK_EQ(TreelitePredictorQueryLeafOutputType(treelite_model_, &output_type), 0)
      << TreeliteGetLastError();
  CHECK_EQ(std::string(output_type), "float32")
      << "Only float32 output types are supported, got " << output_type;
  size_t num_output_class;  // > 1 for multi-class classification; 1 otherwise
  CHECK_EQ(TreelitePredictorQueryNumClass(treelite_model_, &num_output_class), 0)
      << TreeliteGetLastError();
  treelite_output_buffer_size_ = num_output_class;
  treelite_output_.empty();

  // NOTE: second dimension of the output shape is smaller than num_output_class
  //       when a multi-class classifier outputs only the class prediction
  //       (argmax) To detect this edge case, run TreelitePredictorQueryResultSize()
  DMatrixHandle tmp_matrix;
  std::vector<float> tmp_in(treelite_num_feature_);
  const float missing_value = 0.0f;
  CHECK_EQ(TreeliteDMatrixCreateFromMat(tmp_in.data(), "float32", /*num_row=*/1,
                                        treelite_num_feature_, &missing_value, &tmp_matrix),
           0)
      << TreeliteGetLastError();
  CHECK_EQ(TreelitePredictorQueryResultSize(treelite_model_, tmp_matrix, &treelite_output_size_), 0)
      << TreeliteGetLastError();
  CHECK_LE(treelite_output_size_, num_output_class) << "Precondition violated";

  UpdateInputShapes();
  has_sparse_input_ = false;
  if (!paths.metadata.empty() && !IsFileEmpty(paths.metadata)) {
    LoadJsonFromFile(paths.metadata, this->metadata_);
    ValidateDeviceTypeIfExists();
    if (metadata_.count("Model") && metadata_["Model"].count("SparseInput")) {
      has_sparse_input_ = metadata_["Model"]["SparseInput"].get<std::string>() == "1";
    }
  }
}

void TreeliteModel::UpdateInputShapes() {
  input_shapes_.resize(num_inputs_);
  std::vector<int64_t> input_shape(kInputDim);
  input_shape[0] = treelite_input_ ? static_cast<int64_t>(treelite_input_->num_row) : -1;
  input_shape[1] = static_cast<int64_t>(treelite_num_feature_);
  input_shapes_[0] = input_shape;
}

std::vector<std::string> TreeliteModel::GetWeightNames() const {
  throw dmlc::Error("GetWeightNames is not supported by Treelite backend.");
  return std::vector<std::string>();  // unreachable
}

const char* TreeliteModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return INPUT_NAME.c_str();
}
const int TreeliteModel::GetInputDim(int index) const { return kInputDim; }

const int64_t TreeliteModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  const std::vector<int64_t>& shape = GetInputShape(index);
  if (dlr::HasNegative(shape.data(), shape.size())) return -1;
  return abs(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
}

const char* TreeliteModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return INPUT_TYPE.c_str();
}

const char* TreeliteModel::GetWeightName(int index) const {
  throw dmlc::Error("GetWeightName is not supported by Treelite backend.");
  return "";  // unreachable
}

void TreeliteModel::SetInput(const char* name, const int64_t* shape, const void* input, int dim) {
  // NOTE: Assume that missing values are represented by NAN
  CHECK_SHAPE("Mismatch found in input dimension", dim, kInputDim);
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
  float* input_f = (float*)input;

  // NOTE: Assume row-major (C) layout
  treelite_input_->data.reserve(batch_size * num_col);
  treelite_input_->col_ind.reserve(batch_size * num_col);
  treelite_input_->row_ptr.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = 0; j < num_col; ++j) {
      if (std::isnan(input_f[i * num_col + j])) continue;
      if (has_sparse_input_ && input_f[i * num_col + j] == 0.0f) continue;
      treelite_input_->data.push_back(input_f[i * num_col + j]);
      treelite_input_->col_ind.push_back(j);
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
  CHECK_EQ(
      TreeliteDMatrixCreateFromCSR(treelite_input_->data.data(), "float32",
                                   treelite_input_->col_ind.data(), treelite_input_->row_ptr.data(),
                                   batch_size, treelite_num_feature_, &treelite_input_->handle),
      0)
      << TreeliteGetLastError();
  UpdateInputShapes();
}

void TreeliteModel::GetInput(const char* name, void* input) {
  throw dmlc::Error("GetInput is not supported by Treelite backend.");
}

void TreeliteModel::GetOutputShape(int index, int64_t* shape) const {
  // Use -1 if input is yet unspecified and batch size is not known
  shape[0] = treelite_input_ ? static_cast<int64_t>(treelite_input_->num_row) : -1;
  shape[1] = static_cast<int64_t>(treelite_output_size_);
}

void TreeliteModel::GetOutput(int index, void* out) {
  CHECK(treelite_input_);
  std::memcpy(out, treelite_output_.data(),
              sizeof(float) * (treelite_input_->num_row) * treelite_output_size_);
}

const void* TreeliteModel::GetOutputPtr(int index) const {
  CHECK(treelite_input_);
  return treelite_output_.data();
}

void TreeliteModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  if (treelite_input_) {
    *size = static_cast<int64_t>(treelite_input_->num_row * treelite_output_size_);
  } else {
    // Input is yet unspecified and batch is not known
    *size = -1;
  }
  *dim = 2;
}

const char* TreeliteModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return OUTPUT_TYPE.c_str();
}

void TreeliteModel::Run() {
  size_t out_result_size;
  CHECK(treelite_input_);
  treelite_output_.resize(treelite_input_->num_row * treelite_output_buffer_size_);
  CHECK_EQ(TreelitePredictorPredictBatch(treelite_model_, treelite_input_->handle, 0, 0,
                                         (PredictorOutputHandle*)treelite_output_.data(),
                                         &out_result_size),
           0)
      << TreeliteGetLastError();
}

void TreeliteModel::SetNumThreads(int threads) {
  throw dmlc::Error("SetNumThreads is not supported by Treelite backend.");
}

void TreeliteModel::UseCPUAffinity(bool use) {
  throw dmlc::Error("UseCPUAffinity is not supported by Treelite backend.");
}
