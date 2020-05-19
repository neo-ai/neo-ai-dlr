#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "dlr.h"

uint8_t* LoadImageAndPreprocess(const std::string& img_path, size_t size) {
  std::string line;
  std::ifstream fp(img_path);
  uint8_t* img = new uint8_t[size];
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
  EXPECT_STREQ("tflite", backend_name);

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

  // GetDLRInputType
  const char* input_type;
  if (GetDLRInputType(&handle, 0, &input_type)) {
    FAIL() << "GetDLRInputType failed";
  }
  LOG(INFO) << "DLRInputType: " << input_type;
  EXPECT_STREQ("uint8", input_type);

  // GetDLROutputSizeDim
  int64_t out_size;
  int out_dim;
  if (GetDLROutputSizeDim(&handle, 0, &out_size, &out_dim)) {
    FAIL() << "GetDLROutputSizeDim failed";
  }
  LOG(INFO) << "GetDLROutputSizeDim.size: " << out_size;
  LOG(INFO) << "GetDLROutputSizeDim.dim: " << out_dim;
  EXPECT_EQ(1001, out_size);
  EXPECT_EQ(2, out_dim);

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
  const int64_t exp_shape[2] = {1, 1001};
  EXPECT_TRUE(std::equal(std::begin(exp_shape), std::end(exp_shape), shape));

  // GetDLROutputType
  const char* output_type;
  if (GetDLROutputType(&handle, 0, &output_type)) {
    FAIL() << "GetDLROutputType failed";
  }
  LOG(INFO) << "DLROutputType: " << output_type;
  EXPECT_STREQ("uint8", output_type);

  // Load image
  size_t img_size = 224 * 224 * 3;
  uint8_t* img = LoadImageAndPreprocess("cat224-3.txt", img_size);
  LOG(INFO) << "Input sample: " << +img[0] << "," << +img[1] << " ... "
            << +img[img_size - 1];

  // SetDLRInput
  const int64_t in_shape[4] = {1, 224, 224, 3};
  if (SetDLRInput(&handle, input_name, in_shape, (float*)img, 4)) {
    FAIL() << "SetDLRInput failed";
  }
  LOG(INFO) << "SetDLRInput - OK";

  // GetDLRInput
  uint8_t* input2 = new uint8_t[img_size];
  if (GetDLRInput(&handle, input_name, (float*)input2)) {
    FAIL() << "GetDLRInput failed";
  }
  EXPECT_TRUE(std::equal(img, img + img_size, input2));
  LOG(INFO) << "GetDLRInput - OK";

  // RunDLRModel
  if (RunDLRModel(&handle)) {
    FAIL() << "RunDLRModel failed";
  }
  LOG(INFO) << "RunDLRModel - OK";

  // GetDLROutput
  uint8_t* output = new uint8_t[out_size];
  if (GetDLROutput(&handle, 0, (float*)output)) {
    FAIL() << "GetDLROutput failed";
  }
  size_t max_id = ArgMax(output, out_size);
  LOG(INFO) << "ArgMax: " << max_id << ", Prop: " << +output[max_id];
  // TFLite class range is 1-1000 (output size 1001)
  // Imagenet1000 class range is 0-999
  // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
  EXPECT_EQ(282, max_id);  // TFLite 282 maps to Imagenet 281 - tabby, tabby cat
  EXPECT_GE(output[max_id], 150);
  EXPECT_GE(output[283], 80);  // TFLite 283 maps to Imagenet 282 - tiger cat

  // clean up
  delete[] img;
  delete[] input2;
  delete[] output;
}

TEST(TFLite, CreateDLRModelFromTFLiteQuant) {
  // CreateDLRModelFromTFLite (use quant tflite file)
  const char* model_file =
      "./mobilenet_v1_0.75_224_quant/mobilenet_v1_0.75_224_quant.tflite";
  int threads = 2;
  int use_nn_api = 0;

  DLRModelHandle handle = NULL;
  if (CreateDLRModelFromTFLite(&handle, model_file, threads, use_nn_api)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModelFromTFLite - OK";

  CheckAllDLRMethods(handle);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
}

TEST(TFLite, CreateDLRModelQuant) {
  // CreateDLRModel (use folder containing quant tflite file)
  const char* model_dir = "./mobilenet_v1_0.75_224_quant";
  int dev_type = 1;  // 1 - kDLCPU
  int dev_id = 0;

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
