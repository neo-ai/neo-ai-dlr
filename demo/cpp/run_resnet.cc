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

struct NDArray {
  std::vector<float> data;
  std::vector<int64_t> shape;
  int64_t ndim;
  int64_t size;
};

bool is_big_endian();
std::vector<std::vector<float>> RunInference(DLRModelHandle model,
                                             const char* data_path,
                                             const std::string& input_name);

bool is_big_endian() {
  int32_t n = 1;
  // big endian if true
  return (*(char*)&n == 0);
}

/*! \brief A generic inference function using C-API.
 */
std::vector<std::vector<float>> RunInference(DLRModelHandle model,
                                             const char* data_path,
                                             const std::string& input_name) {
  int num_outputs;
  GetDLRNumOutputs(&model, &num_outputs);
  std::vector<int64_t> output_sizes;

  for (int i = 0; i < num_outputs; i++) {
    int64_t cur_size = 0;
    int cur_dim = 0;
    GetDLROutputSizeDim(&model, i, &cur_size, &cur_dim);
    output_sizes.push_back(cur_size);
  }

  std::vector<std::vector<float>> outputs;
  for (auto i : output_sizes) {
    outputs.push_back(std::vector<float>(i, 0));
  }

  std::vector<unsigned long> shape;
  std::vector<double> data;
  npy::LoadArrayFromNumpy(data_path, shape, data);

  NDArray input;
  input.data = std::vector<float>(data.begin(), data.end());
  input.shape = std::vector<int64_t>(shape.begin(), shape.end());
  input.ndim = shape.size();
  input.size = data.size();

  if (SetDLRInput(&model, input_name.c_str(), input.shape.data(),
                  input.data.data(), static_cast<int>(input.ndim)) != 0) {
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

  return outputs;
}

int main(int argc, char** argv) {
  if (is_big_endian()) {
    std::cerr << "Big endian not supported" << std::endl;
    return 1;
  }
  int device_type = 1;
  std::string input_name = "data";
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model dir> <ndarray file> [device] [input name]"
              << std::endl;
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

  std::cout << "Loading model... " << std::endl;
  DLRModelHandle model = NULL;
  if (CreateDLRModel(&model, argv[1], device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }

  std::cout << "Running inference... " << std::endl;
  std::vector<std::vector<float>> outputs =
      RunInference(model, argv[2], input_name);

  int argmax = -1;
  float max_pred = 0.0f;
  for (int i = 0; i < outputs[0].size(); i++) {
    if (outputs[0][i] > max_pred) {
      max_pred = outputs[0][i];
      argmax = i;
    }
  }

  std::cout << "Max probability at " << argmax << " with probability "
            << max_pred << std::endl;

  return 0;
}
