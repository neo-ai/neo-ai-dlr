#include <gtest/gtest.h>
#include "dlr.h"
#include "test_utils.hpp"

DLRModelHandle GetDLRModel() {
  DLRModelHandle model = NULL;
  const char* model_path0 = "./pipeline_model2/sklearn-preproc";
  const char* model_path1 = "./pipeline_model2/xgboost-sm";
  int device_type = 1;  // cpu;
  const char* model_paths[2] = {model_path0, model_path1};
  if (CreateDLRPipeline(&model, 2 /*count*/, model_paths, device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  return model;
}

TEST(PipelineTest, TestGetDLRNumInputs) {
  auto model = GetDLRModel();
  int num_inputs;
  EXPECT_EQ(GetDLRNumInputs(&model, &num_inputs), 0);
  EXPECT_EQ(num_inputs, 1);
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
  EXPECT_STREQ(input_name, "input");
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRInputType) {
  auto model = GetDLRModel();
  const char* input_type;
  EXPECT_EQ(GetDLRInputType(&model, 0, &input_type), 0);
  EXPECT_STREQ(input_type, "json");
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestSetDLRInput) {
  auto model = GetDLRModel();
  std::string in_data = "[[0.5,0.6],[0.55,0.66],[0.73,0.83]]";
  int64_t shape[1] = {static_cast<int64_t>(in_data.length())};
  int ndim = 1;
  const char* input_name = "input";
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, in_data.c_str(), ndim), 0);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRInputShape) {
  auto model = GetDLRModel();
  int64_t input_size;
  int input_dim;
  EXPECT_EQ(GetDLRInputSizeDim(&model, 0, &input_size, &input_dim), 0);
  EXPECT_EQ(input_dim, 2);
  EXPECT_EQ(input_size, -1);
  std::vector<int64_t> shape(input_dim);
  EXPECT_EQ(GetDLRInputShape(&model, 0, shape.data()), 0);
  EXPECT_EQ(shape[0], -1);
  EXPECT_EQ(shape[1], -1);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLROutputShape) {
  auto model = GetDLRModel();
  int64_t output_size;
  int output_dim;
  // output 0
  EXPECT_EQ(GetDLROutputSizeDim(&model, 0, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 2);
  EXPECT_EQ(output_size, -1);
  std::vector<int64_t> shape0(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 0, shape0.data()), 0);
  EXPECT_EQ(shape0[0], -1);
  EXPECT_EQ(shape0[1], 1);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLRNumOutputs) {
  auto model = GetDLRModel();
  int num_outputs;
  EXPECT_EQ(GetDLRNumOutputs(&model, &num_outputs), 0);
  EXPECT_EQ(num_outputs, 1);
  DeleteDLRModel(&model);
}

TEST(PipelineTest, TestGetDLROutputType) {
  auto model = GetDLRModel();
  // output 0
  const char* output_type0;
  EXPECT_EQ(GetDLROutputType(&model, 0, &output_type0), 0);
  EXPECT_STREQ(output_type0, "float32");
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
  std::string in_data = "[[0.3450364,0.3976527],[0.3919772,0.1761725],[0.2091535,0.4263429]]";
  int64_t shape[1] = {static_cast<int64_t>(in_data.length())};
  int ndim = 1;
  const char* input_name = "input";
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, in_data.c_str(), ndim), 0);
  EXPECT_EQ(RunDLRModel(&model), 0);
  // check output metadata
  int64_t output_size;
  int output_dim;
  // output 0 metadata
  EXPECT_EQ(GetDLROutputSizeDim(&model, 0, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 2);
  EXPECT_EQ(output_size, 3);
  std::vector<int64_t> shape0(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 0, shape0.data()), 0);
  EXPECT_EQ(shape0[0], 3);
  EXPECT_EQ(shape0[1], 1);
  const char* output_type0;
  EXPECT_EQ(GetDLROutputType(&model, 0, &output_type0), 0);
  EXPECT_STREQ(output_type0, "float32");
  // output 0 data
  float output0[3];
  EXPECT_EQ(GetDLROutput(&model, 0, output0), 0);
  // batch 0, 1 and 2
  EXPECT_FLOAT_EQ(output0[0], 0.7462196);
  EXPECT_FLOAT_EQ(output0[1], 0.5697872);
  EXPECT_FLOAT_EQ(output0[2], 0.6209581);
  DeleteDLRModel(&model);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
