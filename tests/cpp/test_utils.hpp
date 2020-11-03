#ifndef INCLUDE_TEST_UTILS_HPP
#define INCLUDE_TEST_UTILS_HPP

#include <dmlc/logging.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

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

#endif

