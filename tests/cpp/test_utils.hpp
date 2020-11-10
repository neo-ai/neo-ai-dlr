#ifndef INCLUDE_TEST_UTILS_HPP
#define INCLUDE_TEST_UTILS_HPP

#include <dlpack/dlpack.h>
#include <dmlc/logging.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<float> LoadImageAndPreprocess(const std::string& img_path, size_t size,
                                          int batch_size) {
  std::string line;
  std::ifstream fp(img_path);
  std::vector<float> img(size * batch_size);
  size_t i = 0;
  if (fp.is_open()) {
    while (getline(fp, line) && i < size) {
      int v = std::stoi(line);
      float fv = 2.0f / 255.0f * v - 1.0f;
      img[i++] = fv;
    }
    fp.close();
  }

  EXPECT_EQ(size, i);
  LOG(INFO) << "Image read - OK, float[" << i << "]";

  for (int j = 1; j < batch_size; j++) {
    std::copy_n(img.cbegin(), size, img.begin() + j * size);
  }
  return img;
}

DLTensor GetInputDLTensor() {
  size_t img_size = 224 * 224 * 3;

  int64_t shape[4] = {1, 224, 224, 3};
  DLTensor dltensor;
  dltensor.ctx = {kDLCPU, 0};
  dltensor.ndim = 4;
  dltensor.shape = (int64_t*)malloc(dltensor.ndim * sizeof(int64_t));
  dltensor.strides = 0;
  dltensor.byte_offset = 0;
  dltensor.dtype = {kDLFloat, 32, 1};
  dltensor.data = malloc(img_size * sizeof(float));

  // copy shapes
  for (int i = 0; i < dltensor.ndim; i++) dltensor.shape[i] = shape[i];

  // copy data from file
  std::string line;
  std::ifstream fp("cat224-3.txt");
  size_t i = 0;
  float* ptr = (float*)dltensor.data;
  if (fp.is_open()) {
    while (getline(fp, line) && i < img_size) {
      int v = std::stoi(line);
      float fv = 2.0f / 255.0f * v - 1.0f;
      ptr[i++] = fv;
    }
    fp.close();
  }

  EXPECT_EQ(img_size, i);
  LOG(INFO) << "Image read - OK, float[" << i << "]";

  return dltensor;
}

DLTensor GetOutputDLTensor(int64_t size, int ndim, int64_t* shape, uint8_t dtype) {
  DLTensor dltensor;
  dltensor.ctx = {kDLCPU, 0};
  dltensor.ndim = ndim;
  dltensor.shape = (int64_t*)malloc(dltensor.ndim * sizeof(int64_t));
  dltensor.strides = 0;
  dltensor.byte_offset = 0;
  dltensor.dtype = {dtype, 32, 1};
  dltensor.data = malloc(size * sizeof(float));

  // copy shapes
  for (int i = 0; i < dltensor.ndim; i++) dltensor.shape[i] = shape[i];

  return dltensor;
}

void DeleteDLTensor(DLTensor& dltensor) {
  free(dltensor.shape);
  free(dltensor.data);
}

#endif
