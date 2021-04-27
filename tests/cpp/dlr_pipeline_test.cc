#include <gtest/gtest.h>
#include "dlr.h"
#include "test_utils.hpp"

DLRModelHandle GetDLRModel() {
  DLRModelHandle model = NULL;
  /*
                          --== Test Model Summary ==--
  ________________________________________________________________________________
  Layer (type)                    Output Shape         Param #     Connected to
  ================================================================================
  input_0 (InputLayer)            [(None, 4, 4, 1)]    0
  ________________________________________________________________________________
  input_1 (InputLayer)            [(None, 4, 4, 1)]    0
  ________________________________________________________________________________
  output_0 (Add)                  (None, 4, 4, 1)      0           input_0[0][0]
                                                                   input_1[0][0]
  ________________________________________________________________________________
  output_1 (Multiply)             (None, 4, 4, 1)      0           input_0[0][0]
                                                                   input_1[0][0]
  ================================================================================
  */
  const char* model_path0 = "./pipeline_model1";
  const char* model_path1 = "./pipeline_model1";
  const char* model_path2 = "./pipeline_model1";
  int device_type = 1;  // cpu;
  const char* model_paths[3] = {model_path0, model_path1, model_path2};
  if (CreateDLRPipeline(&model, 3 /*count*/, model_paths, device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  return model;
}

TEST(PipelineTest, TestGetDLRDeviceType) {
  const char* model_path = "./pipeline_model1";
  EXPECT_EQ(GetDLRDeviceType(model_path), -1);
}

TEST(PipelineTest, TestGetDLRNumInputs) {
  auto model = GetDLRModel();
  int num_inputs;
  EXPECT_EQ(GetDLRNumInputs(&model, &num_inputs), 0);
  EXPECT_EQ(num_inputs, 2);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRNumWeights) {
  auto model = GetDLRModel();
  int num_weights;
  EXPECT_EQ(GetDLRNumWeights(&model, &num_weights), 0);
  EXPECT_EQ(num_weights, 0);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRInputName) {
  auto model = GetDLRModel();
  const char* input_name;
  EXPECT_EQ(GetDLRInputName(&model, 0, &input_name), 0);
  EXPECT_STREQ(input_name, "input_0");
  EXPECT_EQ(GetDLRInputName(&model, 1, &input_name), 0);
  EXPECT_STREQ(input_name, "input_1");
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRInputType) {
  auto model = GetDLRModel();
  const char* input_type;
  EXPECT_EQ(GetDLRInputType(&model, 0, &input_type), 0);
  EXPECT_STREQ(input_type, "float32");
  EXPECT_EQ(GetDLRInputType(&model, 1, &input_type), 0);
  EXPECT_STREQ(input_type, "float32");
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestSetDLRInput) {
  auto model = GetDLRModel();
  size_t img_size = 4 * 4;
  std::vector<float> img(img_size, 0.1);
  int64_t shape[4] = {1, 1, 4, 4};
  const char* input_name = "input_0";
  std::vector<float> img2(img_size);
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, img.data(), 4), 0);
  EXPECT_EQ(GetDLRInput(&model, input_name, img2.data()), 0);
  EXPECT_EQ(img, img2);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRInputShape) {
  auto model = GetDLRModel();
  int64_t input_size;
  int input_dim;
  EXPECT_EQ(GetDLRInputSizeDim(&model, 0, &input_size, &input_dim), 0);
  EXPECT_EQ(input_dim, 4);
  EXPECT_EQ(input_size, 16);
  std::vector<int64_t> shape(input_dim);
  EXPECT_EQ(GetDLRInputShape(&model, 0, shape.data()), 0);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 1);
  EXPECT_EQ(shape[2], 4);
  EXPECT_EQ(shape[3], 4);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLROutputShape) {
  auto model = GetDLRModel();
  int64_t output_size;
  int output_dim;
  // output 0
  EXPECT_EQ(GetDLROutputSizeDim(&model, 0, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 4);
  EXPECT_EQ(output_size, 16);
  std::vector<int64_t> shape0(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 0, shape0.data()), 0);
  EXPECT_EQ(shape0[0], 1);
  EXPECT_EQ(shape0[1], 1);
  EXPECT_EQ(shape0[2], 4);
  EXPECT_EQ(shape0[3], 4);
  // output 1
  EXPECT_EQ(GetDLROutputSizeDim(&model, 1, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 4);
  EXPECT_EQ(output_size, 16);
  std::vector<int64_t> shape1(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 1, shape1.data()), 0);
  EXPECT_EQ(shape1[0], 1);
  EXPECT_EQ(shape1[1], 1);
  EXPECT_EQ(shape1[2], 4);
  EXPECT_EQ(shape1[3], 4);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRNumOutputs) {
  auto model = GetDLRModel();
  int num_outputs;
  EXPECT_EQ(GetDLRNumOutputs(&model, &num_outputs), 0);
  EXPECT_EQ(num_outputs, 2);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLROutputType) {
  auto model = GetDLRModel();
  // output 0
  const char* output_type0;
  EXPECT_EQ(GetDLROutputType(&model, 0, &output_type0), 0);
  EXPECT_STREQ(output_type0, "float32");
  // output 1
  const char* output_type1;
  EXPECT_EQ(GetDLROutputType(&model, 1, &output_type1), 0);
  EXPECT_STREQ(output_type1, "float32");
  DeleteDLRModel(&model);
}

TEST(PipelineTest, GetDLRHasMetadata) {
  auto model = GetDLRModel();
  bool has_metadata;
  EXPECT_EQ(GetDLRHasMetadata(&model, &has_metadata), 0);
  EXPECT_FALSE(has_metadata);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRBackend) {
  auto model = GetDLRModel();
  const char* backend;
  EXPECT_EQ(GetDLRBackend(&model, &backend), 0);
  EXPECT_STREQ(backend, "pipeline");
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestRunDLRModel_GetDLROutput) {
  auto model = GetDLRModel();
  size_t img_size = 4 * 4;
  std::vector<float> img0(img_size, 0.1);
  std::vector<float> img1(img_size, 0.3);
  int64_t shape[4] = {1, 1, 4, 4};
  const char* input_name0 = "input_0";
  const char* input_name1 = "input_1";
  EXPECT_EQ(SetDLRInput(&model, input_name0, shape, img0.data(), 4), 0);
  EXPECT_EQ(SetDLRInput(&model, input_name1, shape, img1.data(), 4), 0);
  EXPECT_EQ(RunDLRModel(&model), 0);
  // output 0
  float output0[16];
  EXPECT_EQ(GetDLROutput(&model, 0, output0), 0);
  EXPECT_FLOAT_EQ(output0[0], 0.442);
  float* output0_p;
  EXPECT_EQ(GetDLROutputPtr(&model, 0, (const void**)&output0_p), 0);
  EXPECT_FLOAT_EQ(output0_p[0], 0.442);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(output0_p[i], output0[i]);
  }
  // output 1
  float output1[16];
  EXPECT_EQ(GetDLROutput(&model, 1, output1), 0);
  EXPECT_FLOAT_EQ(output1[0], 0.00516);
  float* output1_p;
  EXPECT_EQ(GetDLROutputPtr(&model, 1, (const void**)&output1_p), 0);
  EXPECT_FLOAT_EQ(output1_p[0], 0.00516);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(output1_p[i], output1[i]);
  }
  DeleteDLRModel(&model);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
