#include <iostream>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>
#include <dlr.h>
#include <libgen.h>

struct NDArray {
  std::vector<float> data;
  std::vector<int64_t> shape;
  int64_t ndim;
  int64_t size;
};

bool is_big_endian();
std::vector<std::vector<float>> RunInference(DLRModelHandle model);

// Based on hcho3's test harness, this executable 
// load and test a simple TVM model that sums 2 pairs of tensors 
// and outputs 2 result tensors.
//    data1 data2     data3 data4
//       \  /             \  /
//      output1          output2

int main(int argc, char** argv) {
  if (is_big_endian()) {
    std::cerr << "Big endian not supported" << std::endl;
    return 1;
  }
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " [model dir]" << std::endl;
    return 1;
  }

  DLRModelHandle model;
  if (CreateDLRModel(&model, argv[1], 1, 0) != 0) {
    std::cout << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }

  std::vector<std::vector<float>> tvm_preds = RunInference(model);
  std::vector<std::vector<float>> golden = {{4.0f, 6.0f}, {13.0f, 15.0f, 17.0f}};
  
  // Compare tvm_preds against preset golden
  CHECK_EQ(tvm_preds.size(), golden.size()) << "Failed sanity check";
  for (size_t j = 0; j < tvm_preds.size(); ++j) {
    for (size_t i = 0; i < tvm_preds[j].size(); ++i)
    CHECK_EQ(tvm_preds[j][i], golden[j][i]) << "Failed sanity check";
  }
  std::cout << "==================" << std::endl;
  std::cout << "All checks passed!" << std::endl;
  std::cout << "==================" << std::endl;
  return 0;
}

bool is_big_endian() {
  union {
    int64_t i;
    char c[4];
  } bint = {0x01020304};

  return bint.c[0] == 1; 
}

std::vector<std::vector<float>> RunInference(DLRModelHandle model) {
  float output1[2], output2[3];
  std::vector<std::vector<float>> out_preds;

  NDArray data1 = { {1.0f, 2.0f}, {2, 1}, 2, 2 };
  NDArray data2 = { {3.0f, 4.0f}, {2, 1}, 2, 2 };
  NDArray data3 = { {5.0f, 6.0f, 7.0f}, {3, 1}, 2, 2 };
  NDArray data4 = { {8.0f, 9.0f, 10.0f}, {3, 1}, 2, 2 };
  if (SetDLRInput(&model, "data1", data1.shape.data(), data1.data.data(), static_cast<int>(data1.ndim)) != 0) {
    throw std::runtime_error("Could not set input 'data1'");
  }
  if (SetDLRInput(&model, "data2", data2.shape.data(), data2.data.data(), static_cast<int>(data2.ndim)) != 0) {
    throw std::runtime_error("Could not set input 'data2'");
  }
  if (SetDLRInput(&model, "data3", data3.shape.data(), data3.data.data(), static_cast<int>(data3.ndim)) != 0) {
    throw std::runtime_error("Could not set input 'data3'");
  }
  if (SetDLRInput(&model, "data4", data4.shape.data(), data4.data.data(), static_cast<int>(data4.ndim)) != 0) {
    throw std::runtime_error("Could not set input 'data4'");
  }

  if (RunDLRModel(&model) != 0) {
    std::cout << DLRGetLastError() << std::endl;  
    throw std::runtime_error("Could not run");
  }
  if (GetDLROutput(&model, 0, output1) != 0) {
    throw std::runtime_error("Could not get output");
  }
  if (GetDLROutput(&model, 1, output2) != 0) {
    throw std::runtime_error("Could not get output");
  }

  out_preds.push_back(std::vector<float> (output1, output1 + 2));
  out_preds.push_back(std::vector<float> (output2, output2 + 3));

  return out_preds;
}
