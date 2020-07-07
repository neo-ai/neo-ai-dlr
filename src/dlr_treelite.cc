#include "dlr_treelite.h"

#include <cmath>
#include <cstring>
#include <fstream>

using namespace dlr;


const std::string TreeliteModel::INPUT_NAME = "data";
const std::string TreeliteModel::INPUT_TYPE = "float32";
const std::string TreeliteModel::OUTPUT_TYPE = "float32";

void TreeliteModel::InitModelArtifact(const std::vector<std::string> &paths) {
  model_artifact_ = {};
  std::vector<std::string> filenames = ListFilesInDirectories(paths);
  for (auto filename : filenames) {
    if (filename != LIBDLR && EndsWith(filename, LIBEXT)) {
      model_artifact_.model_lib = filename;
    } else if (filename == "version.json") {
      model_artifact_.ver_json = filename;
    }
  }
  if (model_artifact_.model_lib.empty()) {
    LOG(INFO) << "No valid Treelite model files found under folder(s):";
    for (auto dir : paths) {
      LOG(INFO) << dir;
    }
    LOG(FATAL);
  }
}

void TreeliteModel::SetupTreeliteModel() {
  // If OMP_NUM_THREADS is set, use it to determine number of threads;
  // if not, use the maximum amount of threads
  const char* val = std::getenv("OMP_NUM_THREADS");
  int num_worker_threads = (val ? std::atoi(val) : -1);
  CHECK_EQ(TreelitePredictorLoad(model_artifact_.model_lib.c_str(), num_worker_threads,
                                 &model_),
           0)
      << TreeliteGetLastError();
}

void TreeliteModel::FetchModelNodesData() {
  CHECK_EQ(
      TreelitePredictorQueryNumFeature(model_, &num_of_input_features_),
      0)
      << TreeliteGetLastError();

  CHECK_EQ(
      TreelitePredictorQueryNumOutputGroup(model_, &output_buffer_size_),
      0)
      << TreeliteGetLastError();

  // NOTE: second dimension of the output shape is smaller than num_output_class
  //       when a multi-class classifier outputs only the class prediction
  //       (argmax) To detect this edge case, run TreelitePredictorPredictInst()
  //       once.
  std::vector<TreelitePredictorEntry> tmp_in(num_of_input_features_);
  std::vector<float> tmp_out(output_buffer_size_);
  CHECK_EQ(TreelitePredictorPredictInst(model_, tmp_in.data(), 0,
                                        tmp_out.data(), &output_size_),
           0)
      << TreeliteGetLastError();
  CHECK_LE(output_size_, output_buffer_size_) << "Precondition violated";
  UpdateInputShapes();
  UpdateOutputShapes();
}

void TreeliteModel::UpdateInputShapes() {
  input_shapes_.resize(num_inputs_);
  std::vector<int64_t> input_shape(2);
  input_shape[0] = input_ ? static_cast<int64_t>(input_->num_row) : -1;
  input_shape[1] = static_cast<int64_t>(num_of_input_features_);
  input_shapes_[0] = input_shape;
}

void TreeliteModel::UpdateOutputShapes() {
  output_shapes_.resize(num_outputs_);
  std::vector<int64_t> output_shape(2);
  // Use -1 if input is yet unspecified and batch size is not known
  output_shape[0] = input_ ? static_cast<int64_t>(input_->num_row) : -1;
  output_shape[1] = static_cast<int64_t>(output_size_);
  output_shapes_[0] = output_shape;
}

const int64_t TreeliteModel::GetInputSize(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  std::vector<int64_t> shape = GetInputShape(index);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}

const int TreeliteModel::GetInputDim(int index) const {
  return 2;
}

const std::string& TreeliteModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return INPUT_NAME;
}

const std::string& TreeliteModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return INPUT_TYPE;
}

const std::vector<int64_t>& TreeliteModel::GetInputShape(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index];
}

const std::string& TreeliteModel::GetWeightName(int index) const {
  LOG(FATAL) << "GetWeightName is not supported by Treelite backend";
}


void TreeliteModel::SetInput(const int index, const int64_t batch_size, void* input) {
  input_.reset(new TreeliteInput);
  CHECK(input_);
  input_->row_ptr.push_back(0);
  float* input_f = (float*) input;

  // NOTE: Assume row-major (C) layout
  for (size_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = 0; j < num_of_input_features_; ++j) {
      if (!std::isnan(input_f[i * num_of_input_features_ + j])) {
        input_->data.push_back(input_f[i * num_of_input_features_ + j]);
        input_->col_ind.push_back(j);
      }
    }
    input_->row_ptr.push_back(input_->data.size());
  }
  // Post conditions for CSR matrix initialization
  CHECK_EQ(input_->data.size(), input_->col_ind.size());
  CHECK_EQ(input_->data.size(), input_->row_ptr.back());
  CHECK_EQ(input_->row_ptr.size(), batch_size + 1);

  // Save dimensions for input
  input_->num_row = static_cast<size_t>(batch_size);
  input_->num_col = num_of_input_features_;

  // Register CSR matrix with Treelite backend
  CHECK_EQ(TreeliteAssembleSparseBatch(
               input_->data.data(), input_->col_ind.data(),
               input_->row_ptr.data(), batch_size,
               num_of_input_features_, &input_->handle),
           0)
      << TreeliteGetLastError();

  // Updated input and output shapes to account for batch size.
  UpdateInputShapes();
  UpdateOutputShapes();
}

void TreeliteModel::SetInput(std::string name, const int64_t batch_size, void* input) {
  SetInput(0, batch_size, input);
}

void TreeliteModel::SetInput(const char* name, const int64_t* shape,
                             void* input, int dim) {
  // NOTE: Assume that missing values are represented by NAN
  CHECK_SHAPE("Mismatch found in input dimension", dim, 2);
  // NOTE: If number of columns is less than num_feature, missing columns
  //       will be automatically padded with missing values
  CHECK_LE(static_cast<size_t>(shape[1]), num_of_input_features_)
      << "ClientError: Mismatch found in input shape at dimension 1. Value "
         "read: "
      << shape[1] << ", Expected: " << num_of_input_features_ << " or less";

  std::string node_name(name);
  SetInput(node_name, *shape, input);
}

void TreeliteModel::GetInput(const char* name, void* input) {
  LOG(FATAL) << "GetInput is not supported by Treelite backend";
}

const std::vector<int64_t>& TreeliteModel::GetOutputShape(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_shapes_[index];
}

void TreeliteModel::GetOutput(int index, void* out) {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  CHECK(input_);
  std::memcpy(
      out, output_.data(),
      sizeof(float) * (input_->num_row) * output_size_);
}

const int64_t TreeliteModel::GetOutputSize(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  if (input_) {
    return static_cast<int64_t>(input_->num_row * output_size_);
        ;
  } else {
    // Input is yet unspecified and batch is not known
    return output_size_;
  }
}

const int TreeliteModel::GetOutputDim(int index) const {
  return 2;
}

const std::string& TreeliteModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return OUTPUT_TYPE;
}

void TreeliteModel::Run() {
  size_t out_result_size;
  CHECK(input_);
  output_.resize(input_->num_row *
                          output_buffer_size_);
  CHECK_EQ(TreelitePredictorPredictBatch(
               model_, input_->handle, 1, 0, 0,
               output_.data(), &out_result_size),
           0)
      << TreeliteGetLastError();
}


void TreeliteModel::SetNumThreads(int threads) {
  LOG(FATAL) << "SetNumThreads is not supported by Treelite backend";
}

void TreeliteModel::UseCPUAffinity(bool use) {
  LOG(FATAL) << "UseCPUAffinity is not supported by Treelite backend";
}
