#include "dlr_relayvm.h"

#include <gtest/gtest.h>
#include "test_utils.hpp"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

class RelayVMTest : public ::testing::Test {
 protected:
  int8_t* img;
  const int batch_size = 1;
  size_t img_size = 512 * 512 * 3;
  const int64_t input_shape[4] = {1, 512, 512, 3};
  const int input_dim = 4;

  dlr::RelayVMModel *model;

  RelayVMTest() {
    const int device_type = kDLCPU;
    const int device_id = 0;
    DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
    std::vector<std::string> paths = {"./ssd_mobilenet_v2"};
    model = new dlr::RelayVMModel(paths, ctx);

    img = new int8_t[img_size];
  }

  ~RelayVMTest() {
    delete model;
    delete img;
  }
};

TEST_F(RelayVMTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(RelayVMTest, TestGetInputName) { EXPECT_STREQ(model->GetInputName(0), "image_tensor"); }

TEST_F(RelayVMTest, TestGetInputType) { EXPECT_STREQ(model->GetInputType(0), "uint8"); }

TEST_F(RelayVMTest, TestSetInput) { EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img, input_dim)); }

TEST_F(RelayVMTest, TestGetNumOutputs) { EXPECT_EQ(model->GetNumOutputs(), 3); }

TEST_F(RelayVMTest, TestGetOutputName) {
  EXPECT_STREQ(model->GetOutputName(0), "detection_scores:0");
  EXPECT_STREQ(model->GetOutputName(1), "detection_classes:0");
  EXPECT_STREQ(model->GetOutputName(2), "num_detections:0");
}

TEST_F(RelayVMTest, TestGetOutputType) {
  EXPECT_STREQ(model->GetOutputType(0), "float32");
  EXPECT_STREQ(model->GetOutputType(1), "float32");
  EXPECT_STREQ(model->GetOutputType(2), "float32");
}

TEST_F(RelayVMTest, TestRun) {
  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img, input_dim));
  model->Run();
}

TEST_F(RelayVMTest, TestGetOutputShape) {
  int64_t output_0_shape[3];
  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img, input_dim));
  EXPECT_NO_THROW(model->Run());
  EXPECT_NO_THROW(model->GetOutputShape(0, output_0_shape));
  EXPECT_EQ(output_0_shape[0], 1);
  EXPECT_EQ(output_0_shape[1], 100);
  EXPECT_EQ(output_0_shape[2], 4);
  int64_t output_1_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(1, output_1_shape));
  EXPECT_EQ(output_1_shape[0], 1);
  EXPECT_EQ(output_1_shape[1], 100);
  int64_t output_2_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(2, output_2_shape));
  EXPECT_EQ(output_2_shape[0], 1);
  EXPECT_EQ(output_2_shape[1], 100);
}

TEST_F(RelayVMTest, TestGetOutputSizeDim) {
  int64_t size;
  int dim;
  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img, input_dim));
  EXPECT_NO_THROW(model->Run());
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 400);
  EXPECT_EQ(dim, 3);
  EXPECT_NO_THROW(model->GetOutputSizeDim(1, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
  EXPECT_NO_THROW(model->GetOutputSizeDim(2, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
}
