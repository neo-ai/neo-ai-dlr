#include "dlr_pipeline.h"

#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>

using namespace dlr;


void PipelineModel::CheckModelsCompatibility(const DLRModelPtr& m0, const DLRModelPtr& m1,
                                             const int m1_id, const bool is_runtime_check) {
  if (!is_runtime_check) {
    CHECK_EQ(m0->GetNumOutputs(), m1->GetNumInputs())
      << "Number of outputs/inputs mismatch between models"
      << m1_id - 1 << " and " << m1_id << std::endl;
  }
  // Check each model input
  for (int j = 0; j < m1->GetNumInputs(); j++) {
    int64_t out_size;
    int out_dim;
    m0->GetOutputSizeDim(j, &out_size, &out_dim);
    if (!is_runtime_check) {
      CHECK_EQ(out_dim, m1->GetInputDim(j))
        << "Number of dimensions mismatch between output/input #" << j << ", models "
        << m1_id - 1 << " and " << m1_id << std::endl;
    }
    const int64_t in_size = m1->GetInputSize(j);
    // Skip the check for dynamic sizes (negative size)
    if (out_size >= 0 && in_size >= 0) {
      CHECK_EQ(out_size, m1->GetInputSize(j))
        << "Size mismatch between output/input #" << j << ", models "
        << m1_id - 1 << " and " << m1_id << std::endl;
    }
    if (!is_runtime_check) {
      CHECK_EQ(strcmp(m0->GetOutputType(j), m1->GetInputType(j)), 0)
        << "Type mismatch between output/input #" << j << ", models "
        << m1_id - 1 << " and " << m1_id << std::endl;
    }
    std::vector<int64_t> in_shape = m1->GetInputShape(j);
    std::vector<int64_t> out_shape(in_shape.size());
    m0->GetOutputShape(j, out_shape.data());
    for (int k = 0; k < in_shape.size(); k++) {
      int64_t in_shape_elem = in_shape[k];
      int64_t out_shape_elem = out_shape[k];
      // Skip the check for dynamic shape elements (-1)
      if (in_shape_elem >= 0 && out_shape_elem >= 0) {
        CHECK_EQ(in_shape_elem, out_shape_elem)
          << "Shape mismatch between output/input #" << j << ", models "
          << m1_id - 1 << " and " << m1_id << std::endl;
      }
    }
  }
}

void PipelineModel::SetupPipelineModel() {
  CHECK_GT(dlr_models_.size(), 0) << "List of models is empty";
  count_ = dlr_models_.size();
  num_inputs_ = dlr_models_[0]->GetNumInputs();
  num_weights_ = dlr_models_[0]->GetNumWeights();
  num_outputs_ = dlr_models_.back()->GetNumOutputs();
  for (int i = 0; i < num_inputs_; i++) {
    input_names_.push_back(dlr_models_[0]->GetInputName(i));
    input_types_.push_back(dlr_models_[0]->GetInputType(i));
    input_shapes_.push_back(dlr_models_[0]->GetInputShape(i));
  }
  // Check previous model outputs and current model inputs compatibility
  for (int i = 1; i < count_; i++) {
    const DLRModelPtr prev_model = dlr_models_[i - 1];
    const DLRModelPtr curr_model = dlr_models_[i];
    CheckModelsCompatibility(prev_model, curr_model, i /*m1_id*/, false /*is_runtime_check*/);
  }
}

std::vector<std::string> PipelineModel::GetWeightNames() const {
  return dlr_models_[0]->GetWeightNames();
}

const char* PipelineModel::GetInputName(int index) const {
  return dlr_models_[0]->GetInputName(index);
}

const char* PipelineModel::GetInputType(int index) const {
  return dlr_models_[0]->GetInputType(index);
}

const int PipelineModel::GetInputDim(int index) const {
  return dlr_models_[0]->GetInputDim(index);
}

const int64_t PipelineModel::GetInputSize(int index) const {
  return dlr_models_[0]->GetInputSize(index);
}

const char* PipelineModel::GetWeightName(int index) const {
  return dlr_models_[0]->GetWeightName(index);
}

void PipelineModel::SetInput(const char* name, const int64_t* shape, const void* input, int dim) {
  dlr_models_[0]->SetInput(name, shape, input, dim);
}

void PipelineModel::GetInput(const char* name, void* input) {
  dlr_models_[0]->GetInput(name, input);
}

void PipelineModel::GetOutputShape(int index, int64_t* shape) const {
  dlr_models_.back()->GetOutputShape(index, shape);
}

void PipelineModel::GetOutput(int index, void* out) {
  dlr_models_.back()->GetOutput(index, out);
}

const void* PipelineModel::GetOutputPtr(int index) const {
  return dlr_models_.back()->GetOutputPtr(index);
}

void PipelineModel::GetOutputSizeDim(int index, int64_t* size, int* dim) {
  dlr_models_.back()->GetOutputSizeDim(index, size, dim);
}

const char* PipelineModel::GetOutputType(int index) const {
  return dlr_models_.back()->GetOutputType(index);
}

void PipelineModel::Run() {
  dlr_models_[0]->Run();
  for (int i = 1; i < count_; i++) {
    const DLRModelPtr prev_model = dlr_models_[i - 1];
    const DLRModelPtr curr_model = dlr_models_[i];
    CheckModelsCompatibility(prev_model, curr_model, i /*m1_id*/, true /*is_runtime_check*/);
    // for each model input
    for (int j = 0; j < curr_model->GetNumInputs(); j++) {
      const char* input_name = curr_model->GetInputName(j);
      // Get output shape of previous output.
      int64_t prev_output_size;
      int prev_output_dim;
      prev_model->GetOutputSizeDim(j, &prev_output_size, &prev_output_dim);
      std::vector<int64_t> prev_output_shape(prev_output_dim, -1);
      prev_model->GetOutputShape(j, prev_output_shape.data());
      const void* prev_model_output = prev_model->GetOutputPtr(j);
      curr_model->SetInput(input_name, prev_output_shape.data(), prev_model_output, prev_output_dim);
    }
    curr_model->Run();
  }
}

const char* PipelineModel::GetBackend() const { return "pipeline"; }

void PipelineModel::SetNumThreads(int threads) {
  // Try to set Number of Threads to pipeline models
  // Ignore the errors in case some of the models do not support this feature.
  for (DLRModelPtr m : dlr_models_) {
    try {
      m->SetNumThreads(threads);
    } catch (dmlc::Error& e) {
      // ignore
    }
  }
}

void PipelineModel::UseCPUAffinity(bool use) {
  // Try to set UseCPUAffinity to pipeline models.
  // Ignore the errors in case some of the models do not support this feature.
  for (DLRModelPtr m : dlr_models_) {
    try {
      m->UseCPUAffinity(use);
    } catch (dmlc::Error& e) {
      // ignore
    }
  }
}

const char* PipelineModel::GetOutputName(const int index) const {
  return dlr_models_.back()->GetOutputName(index);
}

int PipelineModel::GetOutputIndex(const char* name) const {
  return dlr_models_.back()->GetOutputIndex(name);
}

void PipelineModel::GetOutputByName(const char* name, void* out) {
  return dlr_models_.back()->GetOutputByName(name, out);
}
