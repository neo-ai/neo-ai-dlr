#include <dmlc/filesystem.h>
#include <iostream>
#include <fstream>
#include "dlr_common.h"
using namespace dlr;

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

void dlr::LoadJsonFromFile(const std::string& path, nlohmann::json& jsonObject) {
  std::ifstream jsonFile (path);
  jsonFile >> jsonObject;
};

DLRBackend dlr::GetBackend(std::vector<std::string> dir_paths) {
  // Support the case where user provides full path to tflite file.
  if (EndsWith(dir_paths[0], ".tflite")) {
    return DLRBackend::kTFLITE;
  }
  if (EndsWith(dir_paths[0], ".pb")) {
    return DLRBackend::kTENSORFLOW;
  }
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
    } else if (EndsWith(filename, ".tflite")) {
      return DLRBackend::kTFLITE;
    } else if (EndsWith(filename, ".pb")) {
      return DLRBackend::kTENSORFLOW;
    } else if (EndsWith(filename, "_hexagon_model.so")) {
      return DLRBackend::kHEXAGON;
    }
  }
  return DLRBackend::kTREELITE;
}
