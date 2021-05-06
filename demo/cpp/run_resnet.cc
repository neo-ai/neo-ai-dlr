#include <dlr.h>
#include <libgen.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dmlc/logging.h"
#include "npy.hpp"

bool is_big_endian();
template <typename T>
void argmax(int& argmax, T& max_pred);
template <typename T>
void RunInference(DLRModelHandle model, const char* data_path, const std::string& input_name,
                  std::vector<std::vector<T>>& outputs);

bool is_big_endian() {
  int32_t n = 1;
  // big endian if true
  return (*(char*)&n == 0);
}

template <typename T>
void argmax(std::vector<T>& data, int& max_id, T& max_pred) {
  max_id = 0;
  max_pred = 0;
  for (int i = 0; i < data.size(); i++) {
    if (data[i] > max_pred) {
      max_pred = data[i];
      max_id = i;
    }
  }
}

/*! \brief A generic inference function using C-API.
 */
template <typename T>
void RunInference(DLRModelHandle model, const char* data_path, const std::string& input_name,
                  std::vector<std::vector<T>>& outputs) {
  int num_outputs;
  GetDLRNumOutputs(&model, &num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    int64_t cur_size = 0;
    int cur_dim = 0;
    GetDLROutputSizeDim(&model, i, &cur_size, &cur_dim);
    std::vector<T> output(cur_size, 0);
    outputs.push_back(output);
  }

  std::vector<unsigned long> in_shape_ul;
  std::vector<T> in_data;
  bool fortran_order;
  npy::LoadArrayFromNumpy(data_path, in_shape_ul, fortran_order, in_data);

  std::vector<int64_t> in_shape = std::vector<int64_t>(in_shape_ul.begin(), in_shape_ul.end());
  int64_t in_ndim = in_shape.size();

  if (SetDLRInput(&model, input_name.c_str(), in_shape.data(), in_data.data(),
                  static_cast<int>(in_ndim)) != 0) {
    throw std::runtime_error("Could not set input '" + input_name + "'");
  }
  if (RunDLRModel(&model) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not run");
  }
  for (int i = 0; i < num_outputs; i++) {
    if (GetDLROutput(&model, i, outputs[i].data()) != 0) {
      throw std::runtime_error("Could not get output" + std::to_string(i));
    }
  }
}

int main(int argc, char** argv) {
  if (is_big_endian()) {
    std::cerr << "Big endian not supported" << std::endl;
    return 1;
  }
  int device_type = 1;
  std::string input_name = "data";
  std::string input_type = "float32";
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model dir> <ndarray file> [device] [input name] [input type]" << std::endl;
    return 1;
  }
  if (argc >= 4) {
    std::string argv3(argv[3]);
    if (argv3 == "cpu") {
      device_type = 1;
    } else if (argv3 == "gpu") {
      device_type = 2;
    } else if (argv3 == "opencl") {
      device_type = 4;
    } else {
      std::cerr << "Unsupported device type!" << std::endl;
      return 1;
    }
  }
  if (argc >= 5) {
    input_name = argv[4];
  }
  if (argc >= 6) {
    input_type = argv[5];
    if (input_type != "float32" && input_type != "uint8") {
      std::cerr << "Unsupported input type. Use float32 or uint8" << std::endl;
      return 1;
    }
  }

  std::cout << "Loading model... " << std::endl;
  DLRModelHandle model = NULL;
  if (CreateDLRModel(&model, argv[1], device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }

  std::cout << "Running inference... " << std::endl;
  int max_id = -1;
  float max_pred = 0.0f;
  if (input_type == "float32") {
    std::vector<std::vector<float>> outputs;
    RunInference(model, argv[2], input_name, outputs);
    argmax(outputs[0], max_id, max_pred);
  } else if (input_type == "uint8") {
    std::vector<std::vector<uint8_t>> outputs;
    RunInference(model, argv[2], input_name, outputs);
    uint8_t max_pred_uint8 = 0;
    argmax(outputs[0], max_id, max_pred_uint8);
    max_pred = max_pred_uint8;
  }
  std::cout << "Max probability is " << max_pred << " at index " << max_id << std::endl;

  // cleanup
  DeleteDLRModel(&model);
  return 0;
}
