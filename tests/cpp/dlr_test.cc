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

TEST(DLR, TestGetDLRNumInputs) {
  auto model = GetDLRModel();
  int num_inputs;
  EXPECT_EQ(GetDLRNumInputs(&model, &num_inputs), 0);
  EXPECT_EQ(num_inputs, 1);
}

TEST(DLR, TestGetDLRNumWeights) {
  auto model = GetDLRModel();
  int num_weights;
  EXPECT_EQ(GetDLRNumWeights(&model, &num_weights), 0);
  EXPECT_EQ(num_weights, 108);
}

TEST(DLR, TestGetDLRInputName) {
  auto model = GetDLRModel();
  const char* input_name;
  EXPECT_EQ(GetDLRInputName(&model, 0, &input_name), 0);
  EXPECT_STREQ(input_name, "input_tensor");
}

TEST(DLR, TestGetDLRInputType) {
  auto model = GetDLRModel();
  const char* input_type;
  EXPECT_EQ(GetDLRInputType(&model, 0, &input_type), 0);
  EXPECT_STREQ(input_type, "float32");
}

TEST(DLR, TestGetDLRWeightName) {
  auto model = GetDLRModel();
  const char* weight_name;
  EXPECT_EQ(GetDLRWeightName(&model, 0, &weight_name), 0);
  EXPECT_STREQ(weight_name, "p0");
  EXPECT_EQ(GetDLRWeightName(&model, 107, &weight_name), 0);
  EXPECT_STREQ(weight_name, "p99");
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
}

TEST(DLR, TestGetDLROutputShape) {
  auto model = GetDLRModel();

  int64_t output_size;
  int output_dim;
  int index = 0;
  EXPECT_EQ(GetDLROutputSizeDim(&model, 0, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 1);
  
  EXPECT_EQ(GetDLROutputSizeDim(&model, 1, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 2);
}

TEST(DLR, TestGetDLRNumOutputs) {
  auto model = GetDLRModel();
  int num_outputs;
  EXPECT_EQ(GetDLRNumOutputs(&model, &num_outputs), 0);
  EXPECT_EQ(num_outputs, 2);
}


int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
