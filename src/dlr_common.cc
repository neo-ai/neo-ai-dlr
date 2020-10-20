#include "dlr_common.h"

#include <dmlc/filesystem.h>

#include <fstream>
using namespace dlr;

bool dlr::IsFileEmpty(const std::string& filePath) {
  std::ifstream pFile(filePath);
  return pFile.peek() == std::ifstream::traits_type::eof();
}

std::string dlr::GetParentFolder(const std::string& path) {
  size_t found = path.find_last_of("/\\");
  if (found >= 0) {
    return path.substr(0, found);
  }
  return "";
}

std::string dlr::GetBasename(const std::string& path) {
#ifdef _WIN32
  /* remove any trailing backward or forward slashes
     (UNIX does this automatically) */
  std::string path_;
  std::string::size_type tmp = path.find_last_of("/\\");
  if (tmp == path.length() - 1) {
    size_t i = tmp;
    while ((path[i] == '/' || path[i] == '\\') && i >= 0) {
      --i;
    }
    path_ = path.substr(0, i + 1);
  } else {
    path_ = path;
  }
  std::vector<char> fname(path_.length() + 1);
  std::vector<char> ext(path_.length() + 1);
  _splitpath_s(path_.c_str(), NULL, 0, NULL, 0, &fname[0], path_.length() + 1,
               &ext[0], path_.length() + 1);
  return std::string(&fname[0]) + std::string(&ext[0]);
#else
  char* path_ = strdup(path.c_str());
  char* base = basename(path_);
  std::string ret(base);
  free(path_);
  return ret;
#endif
}

void dlr::ListDir(const std::string& dirname, std::vector<std::string>& paths) {
  dmlc::io::URI uri(dirname.c_str());
  dmlc::io::FileSystem* fs = dmlc::io::FileSystem::GetInstance(uri);
  std::vector<dmlc::io::FileInfo> file_list;
  fs->ListDirectory(uri, &file_list);
  for (dmlc::io::FileInfo info : file_list) {
    if (info.type != dmlc::io::FileType::kDirectory) {
      paths.push_back(info.path.name);
    }
  }
}

void dlr::LoadJsonFromFile(const std::string& path,
                           nlohmann::json& jsonObject) {
  std::ifstream jsonFile(path);
  try {
    jsonFile >> jsonObject;
  } catch (nlohmann::json::exception&) {
    jsonObject = nullptr;
  }
}

DLRBackend dlr::GetBackend(std::vector<std::string> dir_paths) {
  // Support the case where user provides full path to hexagon file.
  if (EndsWith(dir_paths[0], "_hexagon_model.so")) {
    return DLRBackend::kHEXAGON;
  }
  // Scan Directory content to guess the backend.
  std::vector<std::string> paths;
  for (auto dir : dir_paths) {
    dlr::ListDir(dir, paths);
  }
  for (auto filename : paths) {
    if (EndsWith(filename, ".params")) {
      return DLRBackend::kTVM;
    } else if (EndsWith(filename, ".ro")) {
      return DLRBackend::kRELAYVM;
    } else if (EndsWith(filename, "_hexagon_model.so")) {
      return DLRBackend::kHEXAGON;
    }
  }
  return DLRBackend::kTREELITE;
}

const std::vector<int64_t>& DLRModel::GetInputShape(int index) const {
  CHECK_LT(index, num_inputs_) << "Input index is out of range.";
  return input_shapes_[index];
}


bool DLRModel::HasMetadata() const { 
  return !this->metadata_.is_null(); 
}

void DLRModel::ValidateDeviceTypeIfExists() {
  DLDeviceType device_type;
  try {
    device_type = GetDeviceTypeFromMetadata();
  } catch (dmlc::Error& e) {
    // Ignore missing metadata file or missing device type.
    return;
  }
  if (device_type != 0 && ctx_.device_type != device_type) {
    std::string msg = "Compiled model requires device type \"";
    msg += GetStringFromDeviceType(ctx_.device_type) + "\" but user gave \"";
    msg += GetStringFromDeviceType(device_type) + "\".";
    throw dmlc::Error(msg);
  }
}

DLDeviceType DLRModel::GetDeviceTypeFromMetadata() const {
  if (!this->HasMetadata()) {
    throw dmlc::Error("No metadata file was found!");
  }
  try {
    const std::string& device_type_string = metadata_.at("Requirements")
        .at("TargetDeviceType")
        .get_ref<const std::string&>();
    return GetDeviceTypeFromString(device_type_string);
  } catch (nlohmann::json::out_of_range& e) {
    throw dmlc::Error("TargetDeviceType was not found in metadata!");
  }
}

DLDeviceType dlr::GetDeviceTypeFromString(const std::string& device_type_string) {
  if (device_type_string == "cpu") {
    return DLDeviceType::kDLCPU;
  } else if (device_type_string == "gpu") {
    return DLDeviceType::kDLGPU;
  } else if (device_type_string == "opencl") {
    return DLDeviceType::kDLOpenCL;
  }
  return DLDeviceType::kDLExtDev;
}

std::string dlr::GetStringFromDeviceType(DLDeviceType device_type) {
  if (device_type == DLDeviceType::kDLCPU) {
    return "cpu";
  } else if (device_type == DLDeviceType::kDLGPU) {
    return "gpu";
  } else if (device_type == DLDeviceType::kDLOpenCL) {
    return "opencl";
  }
  return std::string();
}

std::string dlr::GetMetadataFile(const std::string& dirname) {
  std::vector<std::string> paths_vec;
  ListDir(dirname, paths_vec);
  for (auto filename : paths_vec) {
    if (EndsWith(filename, ".meta")) {
      return filename;
    }
  }
  return std::string();
}

DLDeviceType dlr::GetDeviceTypeFromMetadata(const std::vector<std::string>& model_paths) {
  ModelPath paths;
  std::vector<std::string> paths_vec;
  for (auto dir : model_paths) {
    ListDir(dir, paths_vec);
  }
  std::string metadata_file = "";
  for (auto filename : paths_vec) {
    if (EndsWith(filename, ".meta")) {
      metadata_file = filename;
      break;
    }
  }
  if (metadata_file.empty()) {
    throw dmlc::Error("No metadata file was found!");
  }
  nlohmann::json metadata = nullptr;
  LoadJsonFromFile(metadata_file, metadata); 

  if (metadata.is_null()) {
    throw dmlc::Error("No metadata file was found!");
  }
  try {
    const std::string& device_type_string = metadata.at("Requirements")
        .at("TargetDeviceType")
        .get_ref<const std::string&>();
    return GetDeviceTypeFromString(device_type_string);
  } catch (nlohmann::json::out_of_range& e) {
    throw dmlc::Error("TargetDeviceType was not found in metadata!");
  } 
}

bool dlr::HasNegative(const int64_t* arr, const size_t size) {
  return std::any_of(arr, arr + size, [](int64_t x) {return x < 0;});
}
