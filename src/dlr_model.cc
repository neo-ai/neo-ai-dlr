#include "dlr_model.h"
using namespace dlr;

void DLRModel::LoadMetadataFromModelArtifact() {
  if (!model_artifact_->metadata.empty() &&
      !IsFileEmpty(model_artifact_->metadata)) {
    LOG(INFO) << "Loading metadata file: " << model_artifact_->metadata;
    LoadJsonFromFile(model_artifact_->metadata, metadata);
  } else {
    LOG(INFO) << "No metadata found";
  }
}

const std::string DLRModel::GetBackend() const {
  if (backend_ == DLRBackend::kTVM) {
    return "tvm";
  } else if (backend_ == DLRBackend::kTREELITE) {
    return "treelite";
  } else if (backend_ == DLRBackend::kTREELITE) {
    return "hexagon";
  } else {
    LOG(ERROR) << "Unsupported DLRBackend!";
  }
}

const std::string& DLRModel::GetOutputName(int index) const {
  LOG(ERROR) << "GetOutputName is not supported yet!";
}

int DLRModel::GetOutputIndex(const char* name) const {
  LOG(ERROR) << "GetOutputName is not supported yet!";
}

void DLRModel::GetOutputByName(const char* name, void* out) {
  LOG(ERROR) << "GetOutputByName is not supported yet!";
}

bool DLRModel::HasMetadata() const { return !this->metadata.is_null(); }
