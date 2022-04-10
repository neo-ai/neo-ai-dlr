#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../test_utils.hpp"
#include "dlr.h"

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

void CheckInputShape(DLRModelHandle& handle, const int batch_size) {
  // GetDLRInputSizeDim
  int64_t in_size;
  int in_dim;
  if (GetDLRInputSizeDim(&handle, 0, &in_size, &in_dim)) {
    FAIL() << "GetDLRInputSizeDim failed";
  }
  LOG(INFO) << "GetDLRInputSizeDim.size: " << in_size;
  LOG(INFO) << "GetDLRInputSizeDim.dim: " << in_dim;
  if (batch_size == -1) {
    EXPECT_EQ(-1, in_size);
  } else {
    EXPECT_EQ(batch_size * 224 * 224 * 3, in_size);
  }
  EXPECT_EQ(4, in_dim);

  // GetDLRInputShape
  int64_t shape[in_dim];
  if (GetDLRInputShape(&handle, 0, shape)) {
    FAIL() << "GetDLRInputShape failed";
  }
  std::stringstream ss;
  ss << "GetDLRInputShape: (" << shape[0];
  for (int i = 1; i < in_dim; i++) {
    ss << "," << shape[i];
  }
  ss << ")";
  LOG(INFO) << ss.str();
  const int64_t exp_shape[4] = {batch_size, 224, 224, 3};
  EXPECT_TRUE(std::equal(std::begin(exp_shape), std::end(exp_shape), shape));
}

void CheckAllDLRMethods(DLRModelHandle& handle, const int batch_size, const int prev_batch_size) {
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
  EXPECT_STREQ("inputs", input_name);

  // GetDLROutputName
  const char* output_name;
  if (GetDLROutputName(&handle, 0, &output_name)) {
    FAIL() << "GetDLROutputName failed";
  }
  LOG(INFO) << "DLROutputName: " << output_name;
  EXPECT_STREQ("logits", output_name);

  // GetDLROutputIndex
  int index = -1;
  if (GetDLROutputIndex(&handle, output_name, &index)) {
    FAIL() << "GetDLROutputIndex failed";
  }
  LOG(INFO) << "GetDLROutputIndex: " << index;
  EXPECT_EQ(0, index);

  // GetDLRInputType
  const char* input_type;
  if (GetDLRInputType(&handle, 0, &input_type)) {
    FAIL() << "GetDLRInputType failed";
  }
  LOG(INFO) << "DLRInputType: " << input_type;
  EXPECT_STREQ("1", input_type);

  // GetDLROutputType
  const char* output_type;
  if (GetDLROutputType(&handle, 0, &output_type)) {
    FAIL() << "GetDLROutputType failed";
  }
  LOG(INFO) << "DLROutputType: " << output_type;
  EXPECT_STREQ("1", output_type);

  CheckInputShape(handle, prev_batch_size);

  // Load image
  size_t img_size = 224 * 224 * 3;
  std::vector<float> img = LoadImageAndPreprocess("cat224-3.txt", img_size, batch_size);

  LOG(INFO) << "Input sample: [" << img[0] << "," << img[1] << "..." << img[img_size - 1] << "]...["
            << img[img_size * (batch_size - 1)] << "," << img[img_size * (batch_size - 1) + 1]
            << "..." << img[img_size * batch_size - 1] << "]";
  img_size *= batch_size;

  // SetDLRInput
  const int64_t in_shape[4] = {batch_size, 224, 224, 3};
  if (SetDLRInput(&handle, input_name, in_shape, img.data(), 4)) {
    FAIL() << "SetDLRInput failed";
  }
  LOG(INFO) << "SetDLRInput - OK";

  CheckInputShape(handle, batch_size);

  // GetDLRInput
  std::vector<float> input2(img_size);
  if (GetDLRInput(&handle, input_name, input2.data())) {
    FAIL() << "GetDLRInput failed";
  }
  EXPECT_TRUE(std::equal(img.begin(), img.end(), input2.begin()));
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
  std::vector<float> output(out_size);
  if (GetDLROutput(&handle, 0, output.data())) {
    FAIL() << "GetDLROutput failed";
  }
  LOG(INFO) << "GetDLROutput - OK";
  size_t out_size0 = out_size / batch_size;
  for (int i = 0; i < batch_size; i++) {
    size_t out_offset = i * out_size0;
    size_t max_id = ArgMax(output.data(), out_offset, out_size0);
    LOG(INFO) << "ArgMax: " << max_id << ", Prop: " << output[max_id];
    // Tensorflow class range is 1-1000 (output size 1001)
    // Imagenet1000 class range is 0-999
    // Imagenet 282 - tiger cat
    EXPECT_EQ(out_offset + 282, max_id);
    EXPECT_GE(output[max_id], 0.5f);
    // Imagenet 281 - tabby, tabby cat
    EXPECT_GE(output[out_offset + 281], 0.3f);
  }

  // GetDLROutputByName
  std::vector<float> output_1(out_size);
  if (GetDLROutputByName(&handle, "logits", output_1.data())) {
    FAIL() << "GetDLROutputByName failed";
  }
  EXPECT_TRUE(std::equal(output_1.begin(), output_1.end(), output.begin()));
}

TEST(Tensorflow, CreateDLRModelFromTensorflow2) {
  // CreateDLRModelFromTensorflow2
  const char* model_file = "./imagenet_mobilenet_v2";
  DLR_TF2Config tf2_config = {};
  tf2_config.inter_op_parallelism_threads = 2;
  tf2_config.intra_op_parallelism_threads = 2;

  DLRModelHandle handle = nullptr;
  if (CreateDLRModelFromTensorflow2(&handle, model_file, tf2_config)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModelFromTensorflow2 - OK";

  // batch size 1
  CheckAllDLRMethods(handle, 1, -1);
  // Run the same model for input with batch size 8
  CheckAllDLRMethods(handle, 8, 1);

  // DeleteDLRModel
  DeleteDLRModel(&handle);
}

TEST(Tensorflow, CreateDLRModel) {
  // Use generic CreateDLRModel
  // input and output tensor names will be detected automatically.
  const char* model_file = "./imagenet_mobilenet_v2";
  const int dev_type = 1;  // 1 - kDLCPU
  const int dev_id = 0;

  DLRModelHandle handle = nullptr;
  if (CreateDLRModel(&handle, model_file, dev_type, dev_id)) {
    FAIL() << DLRGetLastError() << std::endl;
  }
  LOG(INFO) << "CreateDLRModel - OK";

  // batch size 2
  CheckAllDLRMethods(handle, 2, -1);
  // Run the same model for input with batch size 8
  CheckAllDLRMethods(handle, 8, 2);

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
