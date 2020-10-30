#include "dlr_data_transform.h"

#include <gtest/gtest.h>

#include "dlr_relayvm.h"
#include "test_utils.hpp"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

void ExpectFloatEq(float value, float expected) {
  // TODO(trevmorr): NanSensitiveFloatEq from googlemock would be useful here.
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(value));
  } else {
    EXPECT_EQ(value, expected);
  }
}

TEST(DLR, DataTransformCategoricalString) {
  dlr::DataTransform transform;
  nlohmann::json metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "ColumnTransform": [
            {
              "Type": "CategoricalString",
              "Map": { "apple": 0, "banana": 1, "7": 2 }
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata, 0));
  // EXPECT_FALSE(transform.HasInputTransform(metadata, 1));

  // User input
  const char* data = R"([["apple"], ["banana"], ["7"], [7], ["walrus"], [-5]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  DLDataType dtype = DLDataType{kDLFloat, 32, 1};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(1);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtype, ctx, &transformed_data));

  std::vector<float> expected_output = {0, 1, 2, -1, -1, -1};
  EXPECT_EQ(transformed_data[0]->ndim, 2);
  EXPECT_EQ(transformed_data[0]->shape[0], 6);
  EXPECT_EQ(transformed_data[0]->shape[1], 1);
  for (size_t i = 0; i < expected_output.size(); ++i) {
    CHECK_EQ(static_cast<float*>(transformed_data[0]->data)[i], expected_output[i])
        << "Output at index " << i;
    ;
  }
}

TEST(DLR, DataTransformCategoricalStringNumericColumn) {
  dlr::DataTransform transform;
  nlohmann::json metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "ColumnTransform": [
            {
              "Type": "float",
              "Map": {}
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata, 0));

  // User input
  const char* data =
      R"([["2.345"], [7], ["7"], [-9.7], ["null"], ["-Inf"], ["NaN"], ["InFinITy"]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  DLDataType dtype = DLDataType{kDLFloat, 32, 1};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(1);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtype, ctx, &transformed_data));
  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> expected_output = {2.345, 7, 7, -9.7, kNan, -kInf, kNan, kInf};
  EXPECT_EQ(transformed_data[0]->ndim, 2);
  EXPECT_EQ(transformed_data[0]->shape[0], 8);
  EXPECT_EQ(transformed_data[0]->shape[1], 1);
  for (size_t i = 0; i < expected_output.size(); ++i) {
    ExpectFloatEq(static_cast<float*>(transformed_data[0]->data)[i], expected_output[i]);
  }
}

TEST(DLR, DataTransformMultipleColumn) {
  dlr::DataTransform transform;
  nlohmann::json metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "ColumnTransform": [
            {
              "Type": "float",
              "Map": {}
            },
            {
              "Type": "CategoricalString",
              "Map": { "apple": 0, "banana": 1, "7": 2 }
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata, 0));

  // User input
  const char* data =
      R"([["2.345", "apple"], [7, "7"]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  DLDataType dtype = DLDataType{kDLFloat, 32, 1};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(2);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtype, ctx, &transformed_data));
  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> expected_output_float = {2.345, kNan, 7, 7};
  std::vector<float> expected_output_string = {-1, 0, -1,
                                               2};  // TODO what should float 7 be ? -1 or 2
  EXPECT_EQ(transformed_data[0]->ndim, 2);
  EXPECT_EQ(transformed_data[0]->shape[0], 2);
  EXPECT_EQ(transformed_data[0]->shape[1], 2);
  EXPECT_EQ(transformed_data[1]->ndim, 2);
  EXPECT_EQ(transformed_data[1]->shape[0], 2);
  EXPECT_EQ(transformed_data[1]->shape[1], 2);
  for (size_t i = 0; i < expected_output_float.size(); ++i) {
    ExpectFloatEq(static_cast<float*>(transformed_data[0]->data)[i], expected_output_float[i]);
  }
  for (size_t i = 0; i < expected_output_string.size(); ++i) {
    ExpectFloatEq(static_cast<float*>(transformed_data[1]->data)[i], expected_output_string[i]);
  }
}

/*
TEST(DLR, RelayVMDataTransformInput) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<std::string> paths = {"./onehotencoder"};
  dlr::RelayVMModel* model = new dlr::RelayVMModel(paths, ctx);

  const char* data = R"([["apple", 1, 7], ["banana", 3, 8], ["squash", 2, 9]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  EXPECT_STREQ(model->GetInputType(0), "json");
  model->SetInput("input", shape.data(), const_cast<char*>(data), 1);
  model->Run();

  int64_t size;
  int dim;
  EXPECT_STREQ(model->GetOutputType(0), "float32");
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 3 * 8);
  EXPECT_EQ(dim, 2);
  int64_t output_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], 3);
  EXPECT_EQ(output_shape[1], 8);

  std::vector<float> expected_output = {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                        1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> output(size, 0);
  EXPECT_NO_THROW(model->GetOutput(0, output.data()));
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(output[i], expected_output[i]) << "Output at index " << i;
  }
  delete model;
}
*/