#include "dlr_model.h"
#include "dlr_tvm.h"
#include "dlr_treelite.h"
#ifdef DLR_HEXAGON
#include "dlr_hexagon/dlr_hexagon.h"
#endif
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


DLRModel *DLRModel::create_model(std::string path, int device_type, int device_id) {
  /* Logic to handle Windows drive letter */
  std::string model_path_string{path};
  std::string special_prefix{""};
  if (model_path_string.length() >= 2 && model_path_string[1] == ':' &&
      std::isalpha(model_path_string[0], std::locale("C"))) {
    // Handle drive letter
    special_prefix = model_path_string.substr(0, 2);
    model_path_string = model_path_string.substr(2);
  }

  std::vector<std::string> paths = dmlc::Split(model_path_string, ':');
  paths[0] = special_prefix + paths[0];
  return create_model(paths, device_type, device_id);
}

DLRModel *DLRModel::create_model(std::vector<std::string> paths, int device_type, int device_id) {
  DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
  return create_model(paths, ctx);
}

DLRModel *DLRModel::create_model(std::vector<std::string> paths, const DLContext &ctx) {
  DLRBackend backend = dlr::GetBackend(paths);
  if (backend == DLRBackend::kTVM) {
    return new TVMModel(paths, ctx);
  }
  else if (backend  == DLRBackend::kTREELITE) {
    return new TreeliteModel(paths, ctx);
  }
  #ifdef DLR_HEXAGON
  else if (backend  == DLRBackend::kHEXAGON) {
    DLContext hexagon_ctx;
    hexagon_ctx.device_type = DLDeviceType::kDLCPU;
    hexagon_ctx.device_id = 0;
    int debug_level = 1;
    return new HexagonModel(paths, hexagon_ctx, debug_level);
  }
  #endif
  else {
    LOG(FATAL) << "Unsupported backend!";
  }
}

const std::string DLRModel::GetBackend() const {
  if (backend_ == DLRBackend::kTVM) {
    return "tvm";
  } else if (backend_ == DLRBackend::kTREELITE) {
    return "treelite";
  } else if (backend_ == DLRBackend::kHEXAGON) {
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

void DLRModel::Run() { LOG(ERROR) << "Not Implemented!"; }

void DLRModel::Run(int batch_size, void** inputs, void** outputs) {
  for (int index = 0; index < num_inputs_; index++) {
    SetInput(index, batch_size, *(inputs + index));
  }
  Run();
  for (int index = 0; index < num_outputs_; index++) {
    GetOutput(index, *(outputs + index));
  }
}
