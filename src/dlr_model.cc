#include "dlr_model.h"
#include "dlr_tvm.h"
#include "dlr_treelite.h"
#ifdef DLR_HEXAGON
#include "dlr_hexagon/dlr_hexagon.h"
#endif
using namespace dlr;

void DLRModel::FetchOutputNamesFromMetadata() {
  output_names_ = {};
  try {
    for(int i = 0; i < num_outputs_; i++) {
      output_names_.push_back(metadata.at("Model").at("Outputs").at(i).at("name").get_ref<const std::string&>());
    }
  } catch (nlohmann::json::out_of_range& e) {
    LOG(ERROR) << e.what();
  }
}

void DLRModel::LoadMetadataFromModelArtifact() {
  if (!model_artifact_->metadata.empty() &&
      !IsFileEmpty(model_artifact_->metadata)) {
    LOG(INFO) << "Loading metadata file: " << model_artifact_->metadata;
    LoadJsonFromFile(model_artifact_->metadata, metadata);
    FetchOutputNamesFromMetadata();
  } else {
    LOG(INFO) << "No metadata found";
  }
}

DLRModel *DLRModel::CreateModel(std::string path, std::string device_type, int device_id) {
  auto device_type_str_to_enum = [&device_type]()-> DLDeviceType {
    if (device_type == "cpu") {
      return DLDeviceType::kDLCPU;
    } else if (device_type == "gpu") {
      return DLDeviceType::kDLGPU;
    } else if (device_type == "cpupinned") {
      return DLDeviceType::kDLCPUPinned;
    } else if (device_type == "opencl") {
      return DLDeviceType::kDLOpenCL;
    } else if (device_type == "vulkan") {
      return DLDeviceType::kDLVulkan;
    } else if (device_type == "metal") {
      return DLDeviceType::kDLMetal;
    } else if (device_type == "vpi") {
      return DLDeviceType::kDLVPI;
    } else if (device_type == "rocm") {
      return DLDeviceType::kDLROCM;
    } else if (device_type == "extdev") {
      return DLDeviceType::kDLExtDev;
    } {
      throw dmlc::Error("Unsupport device type!");
    }
  };
  DLContext ctx = {device_type_str_to_enum(), device_id};
  return CreateModel(path, ctx);
}


DLRModel *DLRModel::CreateModel(std::string path, int device_type, int device_id) {
  DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
  return CreateModel(path, ctx);
}

DLRModel *DLRModel::CreateModel(std::string path, const DLContext &ctx) {
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
  return CreateModel(paths, ctx);
}

DLRModel *DLRModel::CreateModel(std::vector<std::string> paths, int device_type, int device_id) {
  DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
  return CreateModel(paths, ctx);
}

DLRModel *DLRModel::CreateModel(std::vector<std::string> paths, const DLContext &ctx) {
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

const std::string& DLRModel::GetInputName(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_names_[index];
}

const std::vector<std::string>& DLRModel::GetInputNames() const {
  return input_names_;
}

const std::string& DLRModel::GetInputType(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_types_[index];
}

const std::vector<std::string>& DLRModel::GetInputTypes() const {
  return input_types_;
}

const std::vector<int64_t>& DLRModel::GetInputShape(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index];
}

const std::string& DLRModel::GetOutputType(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_types_[index];
}

const std::vector<std::string>& DLRModel::GetOutputTypes() const {
  return output_types_;
}

const std::vector<int64_t>& DLRModel::GetOutputShape(int index) const {
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  return output_shapes_[index];
}

const std::string& DLRModel::GetOutputName(const int index) const {
  if (!HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  CHECK_LT(index, num_outputs_) << "Output index is out of range.";
  CHECK_EQ(output_names_.size(), num_outputs_) << "Output node with index " << index << " was not found in metadata file!";
  return output_names_[index];
}

const std::vector<std::string>& DLRModel::GetOutputNames() const {
  if (!HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  return output_names_;
}

int DLRModel::GetOutputIndex(const char* name) const {
  if (!HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  std::string output_node_name(name);
  for (int i = 0; i < num_outputs_; i++) {
    std::string node_name = GetOutputName(i);
    if (output_node_name == node_name) {
      return i;
    }
  }

  std::string msg = "Couldn't find index for output node";
  msg += " " + std::string{name} + "!";
  throw dmlc::Error(msg);
}

void DLRModel::GetOutputByName(const char* name, void* out) {
  int output_index = GetOutputIndex(name);
  GetOutput(output_index, out);
}

bool DLRModel::HasMetadata() const { return !this->metadata.is_null(); }

void DLRModel::Run() { LOG(ERROR) << "Not Implemented!"; }
