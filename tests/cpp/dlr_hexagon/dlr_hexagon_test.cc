#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "dlr.h"

std::vector<uint8_t> LoadImageAndPreprocess(const std::string& img_path, size_t size) {
  std::string line;
  std::ifstream fp(img_path);
  std::vector<uint8_t> img(size);
  size_t i = 0;
  if (fp.is_open()) {
    while (getline(fp, line) && i < size) {
      int v = std::stoi(line);
      img[i++] = v;
    }
    fp.close();
  }

  EXPECT_EQ(size, i);
  LOG(INFO) << "Image read - OK, uint8_t[" << i << "]";
  return img;
}

int ArgMax(uint8_t* data, int size) {
  int idx = 0;
  uint8_t v = 0;
  for (int i = 0; i < size; i++) {
    uint8_t vi = data[i];
    if (vi > v) {
      idx = i;
      v = vi;
    }
  }
  return idx;
}

void CheckAllDLRMethods(DLRModelHandle& handle) {
  // GetDLRBackend
  const char* backend_name;
  if (GetDLRBackend(&handle, &backend_name)) {
    FAIL() << "GetDLRBackend failed";
  }
  LOG(INFO) << "GetDLRBackend: " << backend_name;
  EXPECT_STREQ("hexagon", backend_name);

  // GetDLRNumInputs
  int num_inputs;
  if (GetDLRNumInputs(&handle, &num_inputs)) {
    FAIL() << "GetDLRNumInputs failed";
  }
  LOG(INFO) << "GetDLRNumInputs: " << num_inputs;
  EXPECT_EQ(1, num_inputs);

  // GetDLRNumOutputs
  int num_outputs;
  if (GetDLRNumOutputs(&handle, &num_outputs)) {
    FAIL() << "GetDLRNumOutputs failed";
  }
  LOG(INFO) << "GetDLRNumOutputs: " << num_outputs;
  EXPECT_EQ(1, num_outputs);

  // GetDLRNumWeights
  int num_weights;
  if (GetDLRNumWeights(&handle, &num_weights)) {
    FAIL() << "GetDLRNumWeights failed";
  }
  LOG(INFO) << "GetDLRNumWeights: " << num_weights;
  EXPECT_EQ(0, num_weights);

  // GetDLRInputName
  const char* input_name;
  if (GetDLRInputName(&handle, 0, &input_name)) {
    FAIL() << "GetDLRInputName failed";
  }
  LOG(INFO) << "DLRInputName: " << input_name;
  EXPECT_STREQ("input", input_name);

  // GetDLROutputSizeDim
  int64_t out_size;
  int out_dim;
  if (GetDLROutputSizeDim(&handle, 0, &out_size, &out_dim)) {
    FAIL() << "GetDLROutputSizeDim failed";
  }
  LOG(INFO) << "GetDLROutputSizeDim.size: " << out_size;
  LOG(INFO) << "GetDLROutputSizeDim.dim: " << out_dim;
  EXPECT_EQ(1001, out_size);
  EXPECT_EQ(4, out_dim);

  // GetDLROutputShape
  int64_t shape[out_dim];
  if (GetDLROutputShape(&handle, 0, shape)) {
    FAIL() << "GetDLROutputShape failed";
  }
  std::stringstream ss;
  ss << "GetDLROutputShape: (" << shape[0];
  for (int i = 1; i < out_dim; i++) {
    ss << "," << shape[i];
  }
  ss << ")";
  LOG(INFO) << ss.str();
  const int64_t exp_shape[4] = {1, 1, 1, 1001};
  EXPECT_TRUE(std::equal(std::begin(exp_shape), std::end(exp_shape), shape));

  // Load image
  size_t img_size = 224 * 224 * 3;
  std::vector<uint8_t> img = LoadImageAndPreprocess("cat224-3.txt", img_size);
  LOG(INFO) << "Input sample: [" << +img[0] << "," << +img[1] << "..." << +img[img_size - 1] << "]";

  // SetDLRInput
  const int64_t in_shape[4] = {1, 224, 224, 3};
  if (SetDLRInput(&handle, input_name, in_shape, img.data(), 4)) {
    FAIL() << "SetDLRInput failed";
  }
  LOG(INFO) << "SetDLRInput - OK";

  // GetDLRInput
  std::vector<uint8_t> input2(img_size);
  if (GetDLRInput(&handle, input_name, input2.data())) {
    FAIL() << "GetDLRInput failed";
  }
  EXPECT_EQ(img, input2);
  LOG(INFO) << "GetDLRInput - OK";

  // RunDLRModel
  if (RunDLRModel(&handle)) {
    FAIL() << "RunDLRModel failed";
  }
  LOG(INFO) << "RunDLRModel - OK";

  // GetDLROutput
  std::vector<uint8_t> output(out_size);
  if (GetDLROutput(&handle, 0, output.data())) {
    FAIL() << "GetDLROutput failed";
  }
  LOG(INFO) << "GetDLROutput - OK";
  size_t max_id = ArgMax(output.data(), out_size);
  LOG(INFO) << "ArgMax: " << max_id << ", Prop: " << +output[max_id];
  // TFLite class range is 1-1000 (output size 1001)
  // Imagenet1000 class range is 0-999
  // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
  EXPECT_EQ(282, max_id);  // TFLite 282 maps to Imagenet 281 - tabby, tabby cat
  EXPECT_GE(output[max_id], 150);
  EXPECT_GE(output[283], 80);  // TFLite 283 maps to Imagenet 282 - tiger cat
}

TEST(Hexagon, CreateDLRModelFromHexagonFromFile) {
  // CreateDLRModelFromHexagon (use _hexagon_model.so file)
  const char* model_file = "./dlr_hexagon_model/mobilenet_v1_0.75_224_quant_hexagon_model.so";
  int debug_level = 0;

  DLRModelHandle handle = NULL;
  if (CreateDLRModelFromHexagon(&handle, model_file, debug_level)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModelFromHexagon - OK";

  CheckAllDLRMethods(handle);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
}

TEST(Hexagon, CreateDLRModelFromDir) {
  // CreateDLRModel (use folder containing _hexagon_model.so file)
  const char* model_dir = "./dlr_hexagon_model";
  // Use undefined number of threads
  const int dev_type = 1;  // 1 - kDLCPU
  const int dev_id = 0;

  DLRModelHandle handle = NULL;
  if (CreateDLRModel(&handle, model_dir, dev_type, dev_id)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModel - OK";

  CheckAllDLRMethods(handle);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
