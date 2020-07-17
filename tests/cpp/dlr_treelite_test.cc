#include "dlr_treelite.h"

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

class TreeliteTest : public ::testing::Test {
 protected:
  float *data;
  const int64_t in_size = 69;
  const int in_dim = 2;
  const int64_t in_shape[2] = {1, 69};
  const int64_t out_size = 1;
  const int out_dim = 2;

  dlr::TreeliteModel *model;

  TreeliteTest() {
    // Setup input data
    data = new float[in_size];
    for (auto i = 0; i < in_size; i++) {
      data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Instantiate model
    const int device_type = 1;
    const int device_id = 0;
    DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
    std::vector<std::string> paths = {"./xgboost_test"};
    model = new dlr::TreeliteModel(paths, ctx);
  }

  ~TreeliteTest() {
    delete model;
    delete data;
  }
};

TEST_F(TreeliteTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(TreeliteTest, TestGetInputName) {
  EXPECT_STREQ(model->GetInputName(0), "data");
}

TEST_F(TreeliteTest, TestGetInputType) {
  EXPECT_STREQ(model->GetInputType(0), "float32");
}

TEST_F(TreeliteTest, TestGetInput) {
  EXPECT_NO_THROW(model->SetInput("data", in_shape, data, in_dim));
  try {
    float* observed_input_data = new float[in_size];
    model->GetInput("data", observed_input_data);
    delete[] observed_input_data;
  } catch (std::exception &e) {
    std::string err_msg{e.what()};
    EXPECT_TRUE(err_msg.find("GetInput is not supported by Treelite backend"));
  }
}

TEST_F(TreeliteTest, TestGetNumOutputs) {
  EXPECT_EQ(model->GetNumOutputs(), 1);
}

TEST_F(TreeliteTest, TestGetOutputType) {
  EXPECT_STREQ(model->GetOutputType(0), "float32");
}

TEST_F(TreeliteTest, TestGetOutputShape) {
  EXPECT_NO_THROW(model->SetInput("data", in_shape, data, in_dim));
  int64_t out_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, out_shape));
  EXPECT_EQ(out_shape[0], 1);
  EXPECT_EQ(out_shape[1], 1);
}

TEST_F(TreeliteTest, TestGetOutputSizeDim) {
  int64_t output_size;
  int output_dim;
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &output_size, &output_dim));
  EXPECT_EQ(output_size, out_size);
  EXPECT_EQ(output_dim, out_dim);
}

TEST_F(TreeliteTest, TestGetOutput) {
  EXPECT_NO_THROW(model->SetInput("data", in_shape, data, in_dim));
  EXPECT_NO_THROW(model->Run());
  float output[1];
  EXPECT_NO_THROW(model->GetOutput(0, output));
}
