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

class RelayVMElemTest : public ::testing::Test {
 protected:
  const int batch_size = 1;
  size_t img_size = 512 * 512 * 3;
  const int64_t input_shape[4] = {1, 512, 512, 3};
  const int input_dim = 4;
  std::vector<int8_t> img{std::vector<int8_t>(img_size)};

  dlr::RelayVMModel* model;

  RelayVMElemTest() {
    const int device_type = kDLCPU;
    const int device_id = 0;
    DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
    std::string ro_file = "./ssd_mobilenet_v1/code.ro";
    std::string so_file = "./ssd_mobilenet_v1/compiled.so";
    std::string meta_file = "./ssd_mobilenet_v1/compiled.meta";
    // load ro file
    std::ifstream relay_ob(ro_file, std::ios::binary);
    std::string code_data((std::istreambuf_iterator<char>(relay_ob)),
                          std::istreambuf_iterator<char>());

    std::vector<DLRModelElem> model_elems = {
        {DLRModelElemType::RELAY_EXEC, nullptr, code_data.data(), code_data.size()},
        {DLRModelElemType::TVM_LIB, so_file.c_str(), nullptr, 0},
        {DLRModelElemType::NEO_METADATA, meta_file.c_str(), nullptr, 0}};
    model = new dlr::RelayVMModel(model_elems, ctx);
  }

  ~RelayVMElemTest() { delete model; }
};

TEST_F(RelayVMElemTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(RelayVMElemTest, TestGetInputName) { EXPECT_STREQ(model->GetInputName(0), "image_tensor"); }

TEST_F(RelayVMElemTest, TestGetInputType) { EXPECT_STREQ(model->GetInputType(0), "uint8"); }

TEST_F(RelayVMElemTest, TestGetInputShape) {
  std::vector<int64_t> in_shape(std::begin(input_shape), std::end(input_shape));
  EXPECT_EQ(model->GetInputShape(0), in_shape);
}

TEST_F(RelayVMElemTest, TestGetInputSize) { EXPECT_EQ(model->GetInputSize(0), 1 * 512 * 512 * 3); }

TEST_F(RelayVMElemTest, TestGetInputDim) { EXPECT_EQ(model->GetInputDim(0), 4); }

TEST_F(RelayVMElemTest, TestSetInput) {
  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img.data(), input_dim));
}

TEST_F(RelayVMElemTest, TestGetNumOutputs) { EXPECT_EQ(model->GetNumOutputs(), 4); }

TEST_F(RelayVMElemTest, TestGetOutputName) {
  EXPECT_STREQ(model->GetOutputName(0), "detection_classes:0");
  EXPECT_STREQ(model->GetOutputName(1), "num_detections:0");
  EXPECT_STREQ(model->GetOutputName(2), "detection_boxes:0");
  EXPECT_STREQ(model->GetOutputName(3), "detection_scores:0");
}

TEST_F(RelayVMElemTest, TestGetOutputType) {
  EXPECT_STREQ(model->GetOutputType(0), "float32");
  EXPECT_STREQ(model->GetOutputType(1), "float32");
  EXPECT_STREQ(model->GetOutputType(2), "float32");
  EXPECT_STREQ(model->GetOutputType(3), "float32");
}

TEST_F(RelayVMElemTest, TestRun) {
  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img.data(), input_dim));
  model->Run();
}

TEST_F(RelayVMElemTest, TestGetOutputShape) {
  int64_t output_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));

  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img.data(), input_dim));
  EXPECT_NO_THROW(model->Run());

  int64_t output_0_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_0_shape));
  EXPECT_EQ(output_0_shape[0], 1);
  EXPECT_EQ(output_0_shape[1], 100);
  int64_t output_1_shape[1];
  EXPECT_NO_THROW(model->GetOutputShape(1, output_1_shape));
  EXPECT_EQ(output_1_shape[0], 1);
  int64_t output_2_shape[3];
  EXPECT_NO_THROW(model->GetOutputShape(2, output_2_shape));
  EXPECT_EQ(output_2_shape[0], 1);
  EXPECT_EQ(output_2_shape[1], 100);
  EXPECT_EQ(output_2_shape[2], 4);
  int64_t output_3_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(3, output_3_shape));
  EXPECT_EQ(output_3_shape[0], 1);
  EXPECT_EQ(output_3_shape[1], 100);
}

TEST_F(RelayVMElemTest, TestGetOutputSizeDim) {
  int64_t size;
  int dim;

  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 100);

  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img.data(), input_dim));
  EXPECT_NO_THROW(model->Run());

  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
  EXPECT_NO_THROW(model->GetOutputSizeDim(1, &size, &dim));
  EXPECT_EQ(size, 1);
  EXPECT_EQ(dim, 1);
  EXPECT_NO_THROW(model->GetOutputSizeDim(2, &size, &dim));
  EXPECT_EQ(size, 400);
  EXPECT_EQ(dim, 3);
  EXPECT_NO_THROW(model->GetOutputSizeDim(3, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
}

TEST_F(RelayVMElemTest, TestGetOutput) {
  EXPECT_NO_THROW(model->SetInput("image_tensor", input_shape, img.data(), input_dim));
  EXPECT_NO_THROW(model->Run());

  float output3[100];
  EXPECT_NO_THROW(model->GetOutput(3, output3));
  float* output3_p;
  EXPECT_NO_THROW(output3_p = (float*)model->GetOutputPtr(3));
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(output3_p[i], output3[i]);
  }
}
