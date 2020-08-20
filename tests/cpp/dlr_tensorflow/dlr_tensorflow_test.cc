#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "dlr.h"
#include "test_utils.hpp"

int ArgMax(float* data, int start, int size) {
  int idx = 0;
  float v = 0.0f;
  for (int i = start; i < start + size; i++) {
    float vi = data[i];
    if (vi > v) {
      idx = i;
      v = vi;
    }
  }
  return idx;
}

void CheckAllDLRMethods(DLRModelHandle& handle, const int batch_size) {
  // GetDLRBackend
  const char* backend_name;
  if (GetDLRBackend(&handle, &backend_name)) {
    FAIL() << "GetDLRBackend failed";
  }
  LOG(INFO) << "GetDLRBackend: " << backend_name;
  EXPECT_STREQ("tensorflow", backend_name);

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
  EXPECT_STREQ("input:0", input_name);

  // Load image
  size_t img_size = 224 * 224 * 3;
  float* img = LoadImageAndPreprocess("cat224-3.txt", img_size, batch_size);
  LOG(INFO) << "Input sample: [" << img[0] << "," << img[1] << "..."
            << img[img_size - 1] << "]...[" << img[img_size * (batch_size - 1)]
            << "," << img[img_size * (batch_size - 1) + 1] << "..."
            << img[img_size * batch_size - 1] << "]";
  img_size *= batch_size;

  // SetDLRInput
  const int64_t in_shape[4] = {batch_size, 224, 224, 3};
  if (SetDLRInput(&handle, input_name, in_shape, img, 4)) {
    FAIL() << "SetDLRInput failed";
  }
  LOG(INFO) << "SetDLRInput - OK";

  // GetDLRInput
  float* input2 = new float[img_size];
  if (GetDLRInput(&handle, input_name, input2)) {
    FAIL() << "GetDLRInput failed";
  }
  EXPECT_TRUE(std::equal(img, img + img_size, input2));
  LOG(INFO) << "GetDLRInput - OK";

  // RunDLRModel
  if (RunDLRModel(&handle)) {
    FAIL() << "RunDLRModel failed";
  }
  LOG(INFO) << "RunDLRModel - OK";

  // GetDLROutputSizeDim
  int64_t out_size;
  int out_dim;
  if (GetDLROutputSizeDim(&handle, 0, &out_size, &out_dim)) {
    FAIL() << "GetDLROutputSizeDim failed";
  }
  LOG(INFO) << "GetDLROutputSizeDim.size: " << out_size;
  LOG(INFO) << "GetDLROutputSizeDim.dim: " << out_dim;
  EXPECT_EQ(1001 * batch_size, out_size);
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
  const int64_t exp_shape[2] = {batch_size, 1001};
  EXPECT_TRUE(std::equal(std::begin(exp_shape), std::end(exp_shape), shape));

  // GetDLROutput (the first and the last item in the batch)
  float* output = new float[out_size];
  if (GetDLROutput(&handle, 0, output)) {
    FAIL() << "GetDLROutput failed";
  }
  LOG(INFO) << "GetDLROutput - OK";
  size_t out_size0 = out_size / batch_size;
  for (int i = 0; i < batch_size; i++) {
    size_t out_offset = i * out_size0;
    size_t max_id = ArgMax(output, out_offset, out_size0);
    LOG(INFO) << "ArgMax: " << max_id << ", Prop: " << output[max_id];
    // Tensorflow class range is 1-1000 (output size 1001)
    // Imagenet1000 class range is 0-999
    // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    // Tensorflow 283 maps to Imagenet 282 - tiger cat
    EXPECT_EQ(out_offset + 283, max_id);
    EXPECT_GE(output[max_id], 0.5f);
    // Tensorflow 282 maps to Imagenet 281 - tabby, tabby cat
    EXPECT_GE(output[out_offset + 282], 0.3f);
  }

  // clean up
  delete[] img;
  delete[] input2;
  delete[] output;
}

TEST(Tensorflow, CreateDLRModelFromTensorflow) {
  // CreateDLRModelFromTensorflow (use .pb file)
  const char* model_file =
      "./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb";
  DLR_TFConfig tf_config = {};
  tf_config.inter_op_parallelism_threads = 2;
  tf_config.intra_op_parallelism_threads = 2;
  int batch_size = 1;
  const int64_t dims[4] = {batch_size, 224, 224, 3};
  const DLR_TFTensorDesc inputs[1] = {{"input:0", dims, 4}};
  const char* outputs[1] = {"MobilenetV1/Predictions/Reshape_1:0"};

  DLRModelHandle handle = NULL;
  if (CreateDLRModelFromTensorflow(&handle, model_file, inputs, 1, outputs, 1,
                                   tf_config)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModelFromTensorflow - OK";

  CheckAllDLRMethods(handle, batch_size);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
}

TEST(Tensorflow, CreateDLRModelFromTensorflowDir) {
  // CreateDLRModelFromTensorflow (use folder containing .pb file)
  const char* model_dir = "./mobilenet_v1_1.0_224";
  // Use undefined number of threads
  DLR_TFConfig tf_config = {};
  int batch_size = 8;
  const int64_t dims[4] = {batch_size, 224, 224, 3};
  const DLR_TFTensorDesc inputs[1] = {{"input:0", dims, 4}};
  const char* outputs[1] = {"MobilenetV1/Predictions/Reshape_1:0"};

  DLRModelHandle handle = NULL;
  if (CreateDLRModelFromTensorflow(&handle, model_dir, inputs, 1, outputs, 1,
                                   tf_config)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModelFromTensorflow - OK";

  CheckAllDLRMethods(handle, batch_size);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
}

TEST(Tensorflow, AutodetectInputsAndOutputs) {
  // Use generic CreateDLRModel
  // input and output tensor names will be detected automatically.
  const char* model_file =
      "./mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb";
  const int batch_size = 1;
  const int dev_type = 1;  // 1 - kDLCPU
  const int dev_id = 0;

  DLRModelHandle handle = NULL;
  if (CreateDLRModel(&handle, model_file, dev_type, dev_id)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModel - OK";

  CheckAllDLRMethods(handle, batch_size);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
  ASSERT_EQ(nullptr, handle);
  // Test that calling DeleteDLRModel again
  // does not crash the program (no segmentation fault)
  DeleteDLRModel(&handle);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
