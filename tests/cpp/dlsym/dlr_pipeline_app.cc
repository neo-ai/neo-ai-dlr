#include <dlfcn.h>
#include <stdint.h>
#include <string>
#include <iostream>
#include <exception>
#include <vector>
#include <fstream>
#include <iterator>
#include <cstring>
#include <sstream>

typedef void* DLRModelHandle;
int (*CreateDLRModel)(DLRModelHandle* handle, const char* model_path, int dev_type, int dev_id);
int (*CreateDLRPipeline)(DLRModelHandle* handle, int num_models, const char** model_paths,
                         int dev_type, int dev_id);
int (*DeleteDLRModel)(DLRModelHandle* handle);
int (*RunDLRModel)(DLRModelHandle* handle);
int (*GetDLRNumInputs)(DLRModelHandle* handle, int* num_inputs);
int (*GetDLRNumWeights)(DLRModelHandle* handle, int* num_weights);
int (*GetDLRInputName)(DLRModelHandle* handle, int index, const char** input_name);
int (*GetDLRInputType)(DLRModelHandle* handle, int index, const char** input_type);
int (*GetDLRWeightName)(DLRModelHandle* handle, int index, const char** weight_name);
int (*SetDLRInput)(DLRModelHandle* handle, const char* name, const int64_t* shape, const void* input,
                   int dim);
int (*GetDLRInput)(DLRModelHandle* handle, const char* name, void* input);
int (*GetDLRInputShape)(DLRModelHandle* handle, int index, int64_t* shape);
int (*GetDLRInputSizeDim)(DLRModelHandle* handle, int index, int64_t* size, int* dim);
int (*GetDLROutputShape)(DLRModelHandle* handle, int index, int64_t* shape);
int (*GetDLROutput)(DLRModelHandle* handle, int index, void* out);
int (*GetDLROutputPtr)(DLRModelHandle* handle, int index, const void** out);
int (*GetDLRNumOutputs)(DLRModelHandle* handle, int* num_outputs);
int (*GetDLROutputSizeDim)(DLRModelHandle* handle, int index, int64_t* size, int* dim);
int (*GetDLROutputType)(DLRModelHandle* handle, int index, const char** output_type);
int (*GetDLRHasMetadata)(DLRModelHandle* handle, bool* has_metadata);
int (*GetDLROutputName)(DLRModelHandle* handle, const int index, const char** name);
int (*GetDLROutputIndex)(DLRModelHandle* handle, const char* name, int* index);
int (*GetDLROutputByName)(DLRModelHandle* handle, const char* name, void* out);
const char* (*DLRGetLastError)();
int (*GetDLRBackend)(DLRModelHandle* handle, const char** name);
int (*GetDLRDeviceType)(const char* model_path);

void InitDLR(const std::string& dlr_path) {
  void* dlr = dlopen(dlr_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (dlr == nullptr) {
    throw std::runtime_error(std::string("Could not load DLR: ") + dlerror());
  }
  *(void**)(&CreateDLRModel) = dlsym(dlr, "CreateDLRModel");
  *(void**)(&CreateDLRPipeline) = dlsym(dlr, "CreateDLRPipeline");
  *(void**)(&DeleteDLRModel) = dlsym(dlr, "DeleteDLRModel");
  *(void**)(&RunDLRModel) = dlsym(dlr, "RunDLRModel");
  *(void**)(&GetDLRNumInputs) = dlsym(dlr, "GetDLRNumInputs");
  *(void**)(&GetDLRNumWeights) = dlsym(dlr, "GetDLRNumWeights");
  *(void**)(&GetDLRInputName) = dlsym(dlr, "GetDLRInputName");
  *(void**)(&GetDLRInputType) = dlsym(dlr, "GetDLRInputType");
  *(void**)(&GetDLRWeightName) = dlsym(dlr, "GetDLRWeightName");
  *(void**)(&SetDLRInput) = dlsym(dlr, "SetDLRInput");
  *(void**)(&GetDLRInput) = dlsym(dlr, "GetDLRInput");
  *(void**)(&GetDLRInputShape) = dlsym(dlr, "GetDLRInputShape");
  *(void**)(&GetDLRInputSizeDim) = dlsym(dlr, "GetDLRInputSizeDim");
  *(void**)(&GetDLROutputShape) = dlsym(dlr, "GetDLROutputShape");
  *(void**)(&GetDLROutput) = dlsym(dlr, "GetDLROutput");
  *(void**)(&GetDLROutputPtr) = dlsym(dlr, "GetDLROutputPtr");
  *(void**)(&GetDLRNumOutputs) = dlsym(dlr, "GetDLRNumOutputs");
  *(void**)(&GetDLROutputSizeDim) = dlsym(dlr, "GetDLROutputSizeDim");
  *(void**)(&GetDLROutputType) = dlsym(dlr, "GetDLROutputType");
  *(void**)(&GetDLRHasMetadata) = dlsym(dlr, "GetDLRHasMetadata");
  *(void**)(&GetDLROutputName) = dlsym(dlr, "GetDLROutputName");
  *(void**)(&GetDLROutputIndex) = dlsym(dlr, "GetDLROutputIndex");
  *(void**)(&GetDLROutputByName) = dlsym(dlr, "GetDLROutputByName");
  *(void**)(&DLRGetLastError) = dlsym(dlr, "DLRGetLastError");
  *(void**)(&GetDLRBackend) = dlsym(dlr, "GetDLRBackend");
  *(void**)(&GetDLRDeviceType) = dlsym(dlr, "GetDLRDeviceType");
}

void ReadInputData(const std::string& input_file, std::vector<std::string>* input_data) {
  std::ifstream infile(input_file);
  std::string line;
  while (std::getline(infile, line)) {
    input_data->push_back(line);
  }
}

void WriteOutputData(const std::string& output_file, const std::vector<std::string>& output_data) {
  std::ofstream outfile(output_file);
  std::ostream_iterator<std::string> output_iterator(outfile, "\n");
  std::copy(output_data.begin(), output_data.end(), output_iterator);
}

std::string GetFloatOutput(DLRModelHandle pipeline) {
  int64_t output_size = 0;
  int output_dim = 0;
  if (GetDLROutputSizeDim(&pipeline, 0, &output_size, &output_dim) != 0) {
    throw std::runtime_error("GetDLROutputSizeDim failed");
  }
  std::vector<float> output(output_size, 0);
  if (GetDLROutput(&pipeline, 0, output.data()) != 0) {
    throw std::runtime_error(std::string("GetDLROutput failed: ") + DLRGetLastError());
  }
  std::stringstream ss;
  ss << "[[";
  for (size_t i = 0; i < output.size(); ++i) {
    ss << output[i];
    if (i != output.size() - 1) ss << ", ";
  }
  ss << "]]";
  return ss.str();
}

std::string GetJsonOutput(DLRModelHandle pipeline) {
  int64_t output_size = 0;
  int output_dim = 0;
  if (GetDLROutputSizeDim(&pipeline, 0, &output_size, &output_dim) != 0) {
    throw std::runtime_error("GetDLROutputSizeDim failed");
  }
  std::vector<char> output(output_size, 0);
  if (GetDLROutput(&pipeline, 0, output.data()) != 0) {
    throw std::runtime_error(std::string("GetDLROutput failed: ") + DLRGetLastError());
  }
  return std::string(output.begin(), output.end());
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " <input data file> <output data file> [model 0 path] [model 1 path] ... [model n path]"
        << std::endl;
    return 1;
  }
  std::string input_file(argv[1]);
  std::string output_file(argv[2]);
  std::vector<const char*> model_paths(argv + 3, argv + argc);

  std::string dlr_path = std::string(model_paths[0]) + "/libdlr.so";
  std::cout << "Using libdlr.so from: " << dlr_path << std::endl;
  InitDLR(dlr_path);

  for (size_t i = 0; i < model_paths.size(); ++i) {
    std::cout << "Pipeline model " << i << ": " << model_paths[i] << std::endl;
  }
  DLRModelHandle pipeline = nullptr;
  if (CreateDLRPipeline(&pipeline, model_paths.size(), model_paths.data(), /*dev_type=*/1, /*dev_id=*/0) != 0) {
    throw std::runtime_error("CreateDLRPipeline failed");
  }

  std::cout << "Using input data from: " << input_file << std::endl;
  std::vector<std::string> input_data;
  ReadInputData(input_file, &input_data);

  std::vector<std::string> output_data;
  for (size_t i = 0; i < input_data.size(); ++i) {
    std::vector<int64_t> input_shape = {static_cast<int64_t>(input_data[i].size())};
    if (SetDLRInput(&pipeline, "input", input_shape.data(), input_data[i].data(), 1) != 0) {
      throw std::runtime_error(std::string("SetDLRInput failed: ") + DLRGetLastError());
    }

    if (RunDLRModel(&pipeline) != 0) {
      throw std::runtime_error(std::string("RunDLRModel failed: ") + DLRGetLastError());
    }

    const char* output_type;
    GetDLROutputType(&pipeline, 0, &output_type);
    if (std::strcmp(output_type, "float32") == 0) {
      output_data.push_back(GetFloatOutput(pipeline));
    } else if (std::strcmp(output_type, "json") == 0) {
      output_data.push_back(GetJsonOutput(pipeline));
    } else {
      throw std::runtime_error("Output type is not json or float32");
    }
  }

  std::cout << "Writing output data to: " << output_file << std::endl;
  WriteOutputData(output_file, output_data);

  DeleteDLRModel(&pipeline);
}
