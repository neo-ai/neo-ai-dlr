#include "dlr_relayvm.h"
#include "dlr_data_transform.h"

#include <gtest/gtest.h>
#include "test_utils.hpp"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

TEST(DLR, DataTransformCategoricalString) {
  dlr::DataTransform transform;
  nlohmann::json metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "0": {
            "CategoricalString": [
              { "apple": 0, "banana": 1, "7": 2 }
            ]
          }
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata, 0));
  EXPECT_FALSE(transform.HasInputTransform(metadata, 1));

  // User input
  const char* data = R"([["apple"], ["banana"], ["7"], [7], ["walrus"], [-5]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  DLDataType dtype = DLDataType{kDLFloat, 32, 1};
  DLContext ctx = DLContext{kDLCPU, 0};
  tvm::runtime::NDArray transformed_data;
  EXPECT_NO_THROW(transformed_data = transform.TransformInput(metadata, 0, shape.data(), const_cast<char*>(data), shape.size(), dtype, ctx));

  std::vector<float> expected_output = {0, 1, 2, 2, -1, -1};
  EXPECT_EQ(transformed_data->ndim, 2);
  EXPECT_EQ(transformed_data->shape[0], 6);
  EXPECT_EQ(transformed_data->shape[1], 1);
  for (size_t i = 0; i < expected_output.size(); ++i) {
    CHECK_EQ(static_cast<float*>(transformed_data->data)[i], expected_output[i]) << i;
  }
}

TEST(DLR, DataTransformCategoricalStringNumericColumn) {
  dlr::DataTransform transform;
  nlohmann::json metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "0": {
            "CategoricalString": [
              {}
            ]
          }
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata, 0));
  // tvm::runtime::NDArray transformed;
  // EXPECT_NO_THROW(transformed, transform.TransformInput());
}

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
