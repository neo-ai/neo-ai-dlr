#include <dlr.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dmlc/logging.h"

/*! \brief Prints model metadata using DLR C-API.
 */
void peek_model(DLRModelHandle model) {
  int num_inputs;
  int num_weights;
  int num_outputs;
  const char* backend;
  std::vector<const char*> input_names;
  std::vector<const char*> weight_names;
  std::vector<std::vector<int64_t>> output_shapes;

  GetDLRBackend(&model, &backend);
  std::cout << "backend is " << backend << std::endl;

  GetDLRNumInputs(&model, &num_inputs);
  std::cout << "num_inputs = " << num_inputs << std::endl;

  GetDLRNumWeights(&model, &num_weights);
  std::cout << "num_weights = " << num_weights << std::endl;

  GetDLRNumOutputs(&model, &num_outputs);
  std::cout << "num_outputs = " << num_outputs << std::endl;

  input_names.resize(num_inputs);
  std::cout << "input_names: ";
  for (int i = 0; i < num_inputs; i++) {
    GetDLRInputName(&model, i, &input_names[i]);
    std::cout << input_names[i] << ", ";
  }
  std::cout << std::endl;

  // weight_names.resize(num_weights);
  // std::cout << "weight_names: ";
  // for (int i = 0; i < num_weights; i++) {
  //   std::cout << " ";
  //   GetDLRWeightName(&model, i, &weight_names[i]);
  //   std::cout << weight_names[i] << ", ";
  // }
  // std::cout << std::endl;

  output_shapes.resize(num_outputs);
  std::cout << "output shapes: " << std::endl;
  for (int i = 0; i < num_outputs; i++) {
    int64_t size = 0;
    int dim = 0;
    GetDLROutputSizeDim(&model, i, &size, &dim);
    output_shapes[i].resize(dim);
    GetDLROutputShape(&model, i, output_shapes[i].data());
    std::cout << "[";
    for (int id = 0; id < dim; id++) {
      std::cout << std::to_string(output_shapes[i][id]) << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

int main(int argc, char** argv) {
  int device_type = 1;
  std::string input_name = "data";
  if (argc < 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " <model dir> [device_type]";
    return 1;
  }
  if (argc >= 3) {
    std::string argv2(argv[2]);
    if (argv2 == "cpu") {
      device_type = 1;
    } else if (argv2 == "gpu") {
      device_type = 2;
    } else if (argv2 == "opencl") {
      device_type = 4;
    } else {
      LOG(FATAL) << "Unsupported device type!";
      return 1;
    }
  }

  DLRModelHandle model = NULL;
  if (CreateDLRModel(&model, argv[1], device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }

  peek_model(model);

  return 0;
}
