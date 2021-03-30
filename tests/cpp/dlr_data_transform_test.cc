#include "dlr_data_transform.h"

#include <gtest/gtest.h>

#include "dlr_relayvm.h"
#include "dlr_tvm.h"
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
              "Map": [{ "apple": 0, "banana": 1, "7": 2 }]
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata));

  // User input
  const char* data = R"([["apple"], ["banana"], ["7"], [7], ["walrus"], [-5]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  std::vector<DLDataType> dtypes = {DLDataType{kDLFloat, 32, 1}};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(1);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtypes, ctx, &transformed_data));
  // Test that same buffer is reused.
  const void* buffer = transformed_data[0]->data;
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtypes, ctx, &transformed_data));
  EXPECT_EQ(buffer, transformed_data[0]->data);

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
              "Type": "Float"
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata));

  // User input
  const char* data =
      R"([["2.345"], [7], ["7"], [-9.7], ["null"], ["-Inf"], ["NaN"], ["InFinITy"]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  std::vector<DLDataType> dtypes = {DLDataType{kDLFloat, 32, 1}};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(1);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtypes, ctx, &transformed_data));
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
              "Type": "Float"
            },
            {
              "Type": "CategoricalString",
              "Map": [
                {},
                {"apple": 0, "banana": 1, "7": 2}
              ]
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata));

  // User input
  const char* data = R"([["2.345", "apple"], [7, "7"]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  // Model input
  std::vector<DLDataType> dtypes = {DLDataType{kDLFloat, 32, 1}, DLDataType{kDLFloat, 32, 1}};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(2);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtypes, ctx, &transformed_data));
  const float kNan = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> expected_output_float = {2.345, kNan, 7, 7};
  std::vector<float> expected_output_string = {2.345, 0, 7, 2};
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

TEST(DLR, DataTransformDateTime) {
  dlr::DataTransform transform;
  nlohmann::json metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "ColumnTransform": [
            {
              "Type": "DateTime", "DateCol": [1]
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata));

  const char* data = R"([["123", "Jan 3th, 2018, 1:34am"]])";
  // ["Feb 11th, 2012, 11:34:59pm"], ["2006-08-23"], ["2017-05-08 14:21:28"], [""]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};

  std::vector<DLDataType> dtypes = {DLDataType{kDLFloat, 32, 1}};
  DLContext ctx = DLContext{kDLCPU, 0};
  std::vector<tvm::runtime::NDArray> transformed_data(1);
  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtypes, ctx, &transformed_data));

  EXPECT_EQ(transformed_data[0]->ndim, 2);
  EXPECT_EQ(transformed_data[0]->shape[0], 1);
  EXPECT_EQ(transformed_data[0]->shape[1], 7);

  std::vector<float> expected_output = {3, 2018, 1, 34, 0, 1, 1};

  for (size_t i = 0; i < expected_output.size(); ++i) {
    ExpectFloatEq(static_cast<float*>(transformed_data[0]->data)[i], expected_output[i]);
  }

  metadata = R"(
    {
      "DataTransform": {
        "Input": {
          "ColumnTransform": [
            {
              "Type": "DateTime", "DateCol": [0, 1]
            }
          ]
        }
      }
    })"_json;
  EXPECT_TRUE(transform.HasInputTransform(metadata));

  data =
      R"([["Feb 11th, 2012, 11:34:59pm", "2006-08-23"], ["2017-05-08 14:21:28", ""], ["12:28:48.000001", "12:28:48.000001+00"], ["2004-09-07 12:28:48.000001-07", "2004-09-07 12:28:48.000001+08"]])";
  shape = {static_cast<int64_t>(std::strlen(data))};

  EXPECT_NO_THROW(transform.TransformInput(metadata, shape.data(), const_cast<char*>(data),
                                           shape.size(), dtypes, ctx, &transformed_data));

  EXPECT_EQ(transformed_data[0]->ndim, 2);
  EXPECT_EQ(transformed_data[0]->shape[0], 4);
  EXPECT_EQ(transformed_data[0]->shape[1], 14);

  expected_output = {6,  2012, 23, 34, 59, 2,  6,  3,  2006, 0,  0,  0,  8,  34,
                     1,  2017, 14, 21, 28, 5,  19, 7,  1900, 0,  0,  0,  1,  52,
                     -1, -1,   12, 28, 48, -1, -1, -1, -1,   12, 28, 48, -1, -1,
                     2,  2004, 12, 28, 48, 9,  37, 2,  2004, 12, 28, 48, 9,  37};

  for (size_t i = 0; i < expected_output.size(); ++i) {
    if (expected_output[i] == -1) continue;
    ExpectFloatEq(static_cast<float*>(transformed_data[0]->data)[i], expected_output[i]);
  }
}

TEST(DLR, RelayVMDataTransformInput) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<std::string> paths = {"./automl"};
  std::vector<std::string> files = dlr::FindFiles(paths);
  dlr::RelayVMModel* model = new dlr::RelayVMModel(files, ctx);
  EXPECT_EQ(model->GetNumInputs(), 1);
  EXPECT_STREQ(model->GetInputType(0), "json");
  EXPECT_STREQ(model->GetInputName(0), "input");

  const char* data = R"([[77, "no", "no", 0, 245.2, 87, 41.68, 254.1, 83, 21.6,239.4, 91, 10.77,
                          7.5, 4, 2.03, 0, 94.77387065117672]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  model->SetInput("input", shape.data(), const_cast<char*>(data), 1);
  model->Run();

  int64_t size;
  int dim;
  EXPECT_STREQ(model->GetOutputType(0), "float32");
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 1 * 608);
  EXPECT_EQ(dim, 2);
  int64_t output_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], 1);
  EXPECT_EQ(output_shape[1], 608);

  // Check first 10 outputs.
  std::vector<float> expected_output = {-0.6220951, -0.58858633, 1.2049465,  -0.67777526,
                                        1.204448,   1.0382348,   -0.8542239, 1.0385356,
                                        0.7477714,  -0.45150468, 0.74640894, -0.9504773};
  std::vector<float> output(size, 0);
  EXPECT_NO_THROW(model->GetOutput(0, output.data()));
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(output[i], expected_output[i]) << "Output at index " << i;
  }
  delete model;
}

TEST(DLR, RelayVMDataTransformOutput) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<std::string> paths = {"./inverselabel"};
  std::vector<std::string> files = dlr::FindFiles(paths);
  dlr::RelayVMModel* model = new dlr::RelayVMModel(files, ctx);

  int64_t size;
  int dim;
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, -1);
  EXPECT_EQ(dim, 1);
  int64_t output_shape[1];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], -1);

  std::vector<int> input_data = {0, 1, 2, 3, -75};
  std::vector<int64_t> shape = {5};
  EXPECT_STREQ(model->GetInputType(0), "int32");
  model->SetInput("input", shape.data(), input_data.data(), shape.size());
  model->Run();

  std::string expected_output =
      "[\"Iris-setosa\",\"Iris-versicolor\",\"Iris-virginica\",\"<unseen_label>\",\"<unseen_"
      "label>\"]";
  EXPECT_STREQ(model->GetOutputType(0), "json");
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, expected_output.size());
  EXPECT_EQ(dim, 1);
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], expected_output.size());

  std::vector<char> output(size, 0);
  EXPECT_NO_THROW(model->GetOutput(0, output.data()));
  std::string output_string(output.begin(), output.end());
  EXPECT_EQ(output_string, expected_output);
  char* output_ptr = nullptr;
  EXPECT_NO_THROW(output_ptr = (char*)model->GetOutputPtr(0));
  std::string output_ptr_string(output_ptr, output_ptr + size);
  EXPECT_EQ(output_ptr_string, expected_output);
  delete model;
}

TEST(DLR, TVMDataTransformInput) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<std::string> paths = {"./automl_static"};
  std::vector<std::string> files = dlr::FindFiles(paths);
  dlr::TVMModel* model = new dlr::TVMModel(files, ctx);
  EXPECT_EQ(model->GetNumInputs(), 1);
  EXPECT_STREQ(model->GetInputType(0), "json");
  EXPECT_STREQ(model->GetInputName(0), "input");

  const char* data = R"([[77, "no", "no", 0, 245.2, 87, 41.68, 254.1, 83, 21.6,239.4, 91, 10.77,
                          7.5, 4, 2.03, 0, 94.77387065117672]])";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  model->SetInput("input", shape.data(), const_cast<char*>(data), 1);
  model->Run();

  int64_t size;
  int dim;
  EXPECT_STREQ(model->GetOutputType(0), "float32");
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 1 * 608);
  EXPECT_EQ(dim, 2);
  int64_t output_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], 1);
  EXPECT_EQ(output_shape[1], 608);

  // Check first 10 outputs.
  std::vector<float> expected_output = {-0.6220951, -0.58858633, 1.2049465,  -0.67777526,
                                        1.204448,   1.0382348,   -0.8542239, 1.0385356,
                                        0.7477714,  -0.45150468, 0.74640894, -0.9504773};
  std::vector<float> output(size, 0);
  EXPECT_NO_THROW(model->GetOutput(0, output.data()));
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(output[i], expected_output[i]) << "Output at index " << i;
  }
  delete model;
}

TEST(DLR, TVMDataTransformOutput) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<std::string> paths = {"./inverselabel_static"};
  std::vector<std::string> files = dlr::FindFiles(paths);
  dlr::TVMModel* model = new dlr::TVMModel(files, ctx);

  int64_t size;
  int dim;
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, -1);
  EXPECT_EQ(dim, 1);
  int64_t output_shape[1];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], -1);

  std::vector<float> input_data = {0, 1, 0.5, 0.4, 0.6};
  std::vector<int64_t> shape = {5};
  EXPECT_STREQ(model->GetInputType(0), "float32");
  model->SetInput("input", shape.data(), input_data.data(), shape.size());
  model->Run();

  std::string expected_output = "[\"False.\",\"True.\",\"False.\",\"False.\",\"True.\"]";
  EXPECT_STREQ(model->GetOutputType(0), "json");
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, expected_output.size());
  EXPECT_EQ(dim, 1);
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], expected_output.size());

  std::vector<char> output(size, 0);
  EXPECT_NO_THROW(model->GetOutput(0, output.data()));
  std::string output_string(output.begin(), output.end());
  EXPECT_EQ(output_string, expected_output);
  char* output_ptr = nullptr;
  EXPECT_NO_THROW(output_ptr = (char*)model->GetOutputPtr(0));
  std::string output_ptr_string(output_ptr, output_ptr + size);
  EXPECT_EQ(output_ptr_string, expected_output);
  delete model;
}
