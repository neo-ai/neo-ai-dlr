#include <gtest/gtest.h>
#include "dlr.h"
#include "test_utils.hpp"

DLRModelHandle GetDLRModel() {
  DLRModelHandle model = nullptr;
  const char* model_path  = "./resnet_v1_5_50";
  int device_type = 1; // cpu;
  if (CreateDLRModel(&model, model_path, device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  return model;
}


TEST(DLR, TestGetDLRDeviceType) {
  const char* model_path  = "./resnet_v1_5_50";
  EXPECT_EQ(GetDLRDeviceType(model_path), -1);
}

TEST(DLR, TestGetDLRNumInputs) {
  auto model = GetDLRModel();
  int num_inputs;
  EXPECT_EQ(GetDLRNumInputs(&model, &num_inputs), 0);
  EXPECT_EQ(num_inputs, 1);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRNumWeights) {
  auto model = GetDLRModel();
  int num_weights;
  EXPECT_EQ(GetDLRNumWeights(&model, &num_weights), 0);
  EXPECT_EQ(num_weights, 108);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRInputName) {
  auto model = GetDLRModel();
  const char* input_name;
  EXPECT_EQ(GetDLRInputName(&model, 0, &input_name), 0);
  EXPECT_STREQ(input_name, "input_tensor");
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRInputType) {
  auto model = GetDLRModel();
  const char* input_type;
  EXPECT_EQ(GetDLRInputType(&model, 0, &input_type), 0);
  EXPECT_STREQ(input_type, "float32");
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRWeightName) {
  auto model = GetDLRModel();
  const char* weight_name;
  EXPECT_EQ(GetDLRWeightName(&model, 0, &weight_name), 0);
  EXPECT_STREQ(weight_name, "p0");
  EXPECT_EQ(GetDLRWeightName(&model, 107, &weight_name), 0);
  EXPECT_STREQ(weight_name, "p99");
  DeleteDLRModel(&model);
}

TEST(DLR, TestSetDLRInput) {
  auto model = GetDLRModel();
  size_t img_size = 224*224*3;
  float* img = LoadImageAndPreprocess("cat224-3.txt", img_size, 1);
  int64_t shape[4] = {1, 224, 224, 3};
  const char* input_name = "input_tensor";
  float* in_img = (float*) malloc(sizeof(float)*224*224*3);
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, img, 4), 0);
  EXPECT_EQ(GetDLRInput(&model, input_name, in_img), 0);
  EXPECT_EQ(*img, *in_img);
  EXPECT_EQ(*(img + 224*224), *(in_img + 224*224));
  EXPECT_EQ(*(img + 224*224*3 - 1), *(in_img + 224*224*3 - 1));
  free(in_img);
  delete [] img;
  DeleteDLRModel(&model);
}


TEST(DLR, TestGetDLRInputShape) {
  auto model = GetDLRModel();
  int64_t input_size;
  int input_dim;
  int index = 0;
  EXPECT_EQ(GetDLRInputSizeDim(&model, 0, &input_size, &input_dim), 0);
  EXPECT_EQ(input_dim, 4);
  EXPECT_EQ(input_size, 1*224*224*3);
  std::vector<int64_t> shape(input_dim);
  EXPECT_EQ(GetDLRInputShape(&model, 0, shape.data()), 0);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 224);
  EXPECT_EQ(shape[2], 224);
  EXPECT_EQ(shape[3], 3);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputShape) {
  auto model = GetDLRModel();
  int64_t output_size;
  int output_dim;
  int index = 0;
  // output 0
  EXPECT_EQ(GetDLROutputSizeDim(&model, 0, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 1);
  EXPECT_EQ(output_size, 1);
  std::vector<int64_t> shape0(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 0, shape0.data()), 0);
  EXPECT_EQ(shape0[0], 1);
  // output 1
  EXPECT_EQ(GetDLROutputSizeDim(&model, 1, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 2);
  EXPECT_EQ(output_size, 1001);
  std::vector<int64_t> shape1(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 1, shape1.data()), 0);
  EXPECT_EQ(shape1[0], 1);
  EXPECT_EQ(shape1[1], 1001);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRNumOutputs) {
  auto model = GetDLRModel();
  int num_outputs;
  EXPECT_EQ(GetDLRNumOutputs(&model, &num_outputs), 0);
  EXPECT_EQ(num_outputs, 2);
}

TEST(DLR, TestGetDLROutputType) {
  auto model = GetDLRModel();
  // output 0
  const char* output_type0;
  EXPECT_EQ(GetDLROutputType(&model, 0, &output_type0), 0);
  EXPECT_STREQ(output_type0, "int32");
  // output 1
  const char* output_type1;
  EXPECT_EQ(GetDLROutputType(&model, 1, &output_type1), 0);
  EXPECT_STREQ(output_type1, "float32");
  DeleteDLRModel(&model);
}

TEST(DLR, GetDLRHasMetadata) {
  auto model = GetDLRModel();
  bool has_metadata;
  EXPECT_EQ(GetDLRHasMetadata(&model, &has_metadata), 0);
  EXPECT_TRUE(has_metadata);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputName) {
  auto model = GetDLRModel();
  // output 0
  const char* output_name0;
  EXPECT_EQ(GetDLROutputName(&model, 0, &output_name0), 0);
  EXPECT_STREQ(output_name0, "ArgMax:0");
  // output 1
  const char* output_name1;
  EXPECT_EQ(GetDLROutputName(&model, 1, &output_name1), 0);
  EXPECT_STREQ(output_name1, "softmax_tensor:0");
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputIndex) {
  auto model = GetDLRModel();
  // output 0
  int output_index0;
  EXPECT_EQ(GetDLROutputIndex(&model, "ArgMax:0", &output_index0), 0);
  EXPECT_EQ(output_index0, 0);
  // output 1
  int output_index1;
  EXPECT_EQ(GetDLROutputIndex(&model, "softmax_tensor:0", &output_index1), 0);
  EXPECT_EQ(output_index1, 1);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRBackend) {
  auto model = GetDLRModel();
  const char* backend;
  EXPECT_EQ(GetDLRBackend(&model, &backend), 0);
  EXPECT_STREQ(backend, "tvm");
  DeleteDLRModel(&model);
}

TEST(DLR, TestRunDLRModel_GetDLROutput) {
  auto model = GetDLRModel();
  size_t img_size = 224*224*3;
  float* img = LoadImageAndPreprocess("cat224-3.txt", img_size, 1);
  int64_t shape[4] = {1, 224, 224, 3};
  const char* input_name = "input_tensor";
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, img, 4), 0);
  EXPECT_EQ(RunDLRModel(&model), 0);
  // output 0
  int output0[1];
  EXPECT_EQ(GetDLROutput(&model, 0, output0), 0);
  EXPECT_EQ(output0[0], 112);
  EXPECT_EQ(GetDLROutputByName(&model, "ArgMax:0", output0), 0);
  EXPECT_EQ(output0[0], 112);
  // output 1
  float output1[1001];
  EXPECT_EQ(GetDLROutput(&model, 1, output1), 0);
  EXPECT_GT(output1[112], 0.01);
  EXPECT_EQ(GetDLROutputByName(&model, "softmax_tensor:0", output1), 0);
  EXPECT_GT(output1[112], 0.01);
  delete [] img;
  DeleteDLRModel(&model);
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
