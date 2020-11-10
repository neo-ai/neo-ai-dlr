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

DLTensor GetInputDLTensor(int ndim, int64_t* shape, const char* filename) {
  size_t img_size = 1;
  for(int i=0; i<ndim; i++) img_size *= shape[i];

  DLTensor dltensor;
  dltensor.ctx = {kDLCPU, 0};
  dltensor.ndim = ndim;
  dltensor.shape = (int64_t*)malloc(ndim * sizeof(int64_t));
  dltensor.strides = 0;
  dltensor.byte_offset = 0;
  dltensor.dtype = {kDLFloat, 32, 1};
  dltensor.data = malloc(img_size * sizeof(float));

  // copy shapes
  for (int i = 0; i < ndim; i++) dltensor.shape[i] = shape[i];

  // copy data from file
  std::string line;
  std::ifstream fp(filename);
  size_t i = 0;
  float* ptr = (float*)dltensor.data;
  EXPECT_EQ(fp.is_open(), true);
  while (getline(fp, line) && i < img_size) {
    int v = std::stoi(line);
    float fv = 2.0f / 255.0f * v - 1.0f;
    ptr[i++] = fv;
  }
  fp.close();

  EXPECT_EQ(img_size, i);
  LOG(INFO) << "Image read - OK, float[" << i << "]";

  return dltensor;
}

DLTensor GetEmptyDLTensor(int ndim, int64_t* shape, uint8_t dtype, uint8_t bits) {
  int64_t size = 1;
  for(int i=0; i<ndim; i++) size *= shape[i];
  
  DLTensor dltensor;
  dltensor.ctx = {kDLCPU, 0};
  dltensor.ndim = ndim;
  dltensor.shape = (int64_t*)malloc(dltensor.ndim * sizeof(int64_t));
  dltensor.strides = 0;
  dltensor.byte_offset = 0;
  dltensor.dtype = {dtype, bits, 1};
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
