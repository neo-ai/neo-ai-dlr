#include "dlr_relayvm.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "test_utils.hpp"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

class RelayVMTest : public ::testing::Test {
 protected:
  const int batch_size = 1;
  size_t img_size = 640 * 640 * 3;
  const int64_t input_shape[4] = {1, 640, 640, 3};
  const int input_dim = 4;
  std::vector<int8_t> img{std::vector<int8_t>(img_size)};

  dlr::RelayVMModel* model;

  RelayVMTest() {
    const int device_type = kDLCPU;
    const int device_id = 0;
    DLDevice dev = {static_cast<DLDeviceType>(device_type), device_id};
    std::vector<std::string> paths = {"./ssd_mobilenet_v1"};
    std::vector<std::string> files = dlr::FindFiles(paths);
    model = new dlr::RelayVMModel(files, dev);
  }

  ~RelayVMTest() { delete model; }
};

TEST_F(RelayVMTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(RelayVMTest, TestGetInputName) { EXPECT_STREQ(model->GetInputName(0), "input_tensor"); }

TEST_F(RelayVMTest, TestGetInputType) { EXPECT_STREQ(model->GetInputType(0), "uint8"); }

TEST_F(RelayVMTest, TestGetInputShape) {
  std::vector<int64_t> in_shape(std::begin(input_shape), std::end(input_shape));
  EXPECT_EQ(model->GetInputShape(0), in_shape);
}

TEST_F(RelayVMTest, TestGetInputSize) { EXPECT_EQ(model->GetInputSize(0), 1 * 640 * 640 * 3); }

TEST_F(RelayVMTest, TestGetInputDim) { EXPECT_EQ(model->GetInputDim(0), 4); }

TEST_F(RelayVMTest, TestSetInput) {
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  // Second time should reuse same buffer.
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
}

TEST_F(RelayVMTest, TestGetNumOutputs) { EXPECT_EQ(model->GetNumOutputs(), 8); }

TEST_F(RelayVMTest, TestGetOutputName) {
  EXPECT_STREQ(model->GetOutputName(0), "detection_anchor_indices");
  EXPECT_STREQ(model->GetOutputName(1), "detection_boxes");
  EXPECT_STREQ(model->GetOutputName(2), "detection_classes");
  EXPECT_STREQ(model->GetOutputName(3), "detection_multiclass_scores");
  EXPECT_STREQ(model->GetOutputName(4), "detection_scores");
  EXPECT_STREQ(model->GetOutputName(5), "num_detections");
  EXPECT_STREQ(model->GetOutputName(6), "raw_detection_boxes");
  EXPECT_STREQ(model->GetOutputName(7), "raw_detection_scores");
}

TEST_F(RelayVMTest, TestGetOutputType) {
  EXPECT_STREQ(model->GetOutputType(0), "float32");
  EXPECT_STREQ(model->GetOutputType(1), "float32");
  EXPECT_STREQ(model->GetOutputType(2), "float32");
  EXPECT_STREQ(model->GetOutputType(3), "float32");
  EXPECT_STREQ(model->GetOutputType(4), "float32");
  EXPECT_STREQ(model->GetOutputType(5), "float32");
  EXPECT_STREQ(model->GetOutputType(6), "float32");
  EXPECT_STREQ(model->GetOutputType(7), "float32");
}

TEST_F(RelayVMTest, TestRun) {
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  model->Run();
}

TEST_F(RelayVMTest, TestGetOutputShape) {
  int64_t output_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));

  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  EXPECT_NO_THROW(model->Run());

  int64_t output_0_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_0_shape));
  EXPECT_EQ(output_0_shape[0], 1);
  EXPECT_EQ(output_0_shape[1], 100);
  int64_t output_1_shape[3];
  EXPECT_NO_THROW(model->GetOutputShape(1, output_1_shape));
  EXPECT_EQ(output_1_shape[0], 1);
  EXPECT_EQ(output_1_shape[1], 100);
  EXPECT_EQ(output_1_shape[2], 4);
  int64_t output_2_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(2, output_2_shape));
  EXPECT_EQ(output_2_shape[0], 1);
  EXPECT_EQ(output_2_shape[1], 100);
  int64_t output_3_shape[3];
  EXPECT_NO_THROW(model->GetOutputShape(3, output_3_shape));
  EXPECT_EQ(output_3_shape[0], 1);
  EXPECT_EQ(output_3_shape[1], 100);
  EXPECT_EQ(output_3_shape[2], 91);
  int64_t output_4_shape[3];
  EXPECT_NO_THROW(model->GetOutputShape(4, output_4_shape));
  EXPECT_EQ(output_4_shape[0], 1);
  EXPECT_EQ(output_4_shape[1], 100);
  int64_t output_5_shape[1];
  EXPECT_NO_THROW(model->GetOutputShape(5, output_5_shape));
  EXPECT_EQ(output_5_shape[0], 1);
  int64_t output_6_shape[3];
  EXPECT_NO_THROW(model->GetOutputShape(6, output_6_shape));
  EXPECT_EQ(output_6_shape[0], 1);
  EXPECT_EQ(output_6_shape[1], 51150);
  EXPECT_EQ(output_6_shape[2], 4);
  int64_t output_7_shape[3];
  EXPECT_NO_THROW(model->GetOutputShape(7, output_7_shape));
  EXPECT_EQ(output_7_shape[0], 1);
  EXPECT_EQ(output_7_shape[1], 51150);
  EXPECT_EQ(output_7_shape[2], 91);
}

TEST_F(RelayVMTest, TestGetOutputSizeDim) {
 int64_t size;
  int dim;

  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  EXPECT_NO_THROW(model->Run());

  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
  EXPECT_NO_THROW(model->GetOutputSizeDim(1, &size, &dim));
  EXPECT_EQ(size, 400);
  EXPECT_EQ(dim, 3);
  EXPECT_NO_THROW(model->GetOutputSizeDim(2, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
  EXPECT_NO_THROW(model->GetOutputSizeDim(3, &size, &dim));
  EXPECT_EQ(size, 9100);
  EXPECT_EQ(dim, 3);
  EXPECT_NO_THROW(model->GetOutputSizeDim(4, &size, &dim));
  EXPECT_EQ(size, 100);
  EXPECT_EQ(dim, 2);
  EXPECT_NO_THROW(model->GetOutputSizeDim(5, &size, &dim));
  EXPECT_EQ(size, 1);
  EXPECT_EQ(dim, 1);
  EXPECT_NO_THROW(model->GetOutputSizeDim(6, &size, &dim));
  EXPECT_EQ(size, 51150*4);
  EXPECT_EQ(dim, 3);
  EXPECT_NO_THROW(model->GetOutputSizeDim(7, &size, &dim));
  EXPECT_EQ(size, 51150*91);
  EXPECT_EQ(dim, 3);
}

TEST_F(RelayVMTest, TestGetOutput) {
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  EXPECT_NO_THROW(model->Run());

  float output2[100];
  EXPECT_NO_THROW(model->GetOutput(2, output2));
  float* output2_p;
  EXPECT_NO_THROW(output2_p = (float*)model->GetOutputPtr(2));
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(output2_p[i], output2[i]);
  }
}

TEST(DLR, TestRelayVMAllocatorDefault) {
  DLDevice dev = {static_cast<DLDeviceType>(kDLCPU), 0};
  std::vector<std::string> paths = {"./ssd_mobilenet_v1"};
  std::vector<std::string> files = dlr::FindFiles(paths);
  dlr::RelayVMModel* model = new dlr::RelayVMModel(files, dev);

  EXPECT_EQ(model->GetAllocatorType(), tvm::runtime::vm::AllocatorType::kPooled);

  delete model;
}

TEST(DLR, TestRelayVMAllocatorEnvVar) {
  EXPECT_EQ(SetEnv("DLR_RELAYVM_ALLOCATOR", "naive"), 0);
  DLDevice dev = {static_cast<DLDeviceType>(kDLCPU), 0};
  std::vector<std::string> paths = {"./ssd_mobilenet_v1"};
  std::vector<std::string> files = dlr::FindFiles(paths);
  dlr::RelayVMModel* model = new dlr::RelayVMModel(files, dev);

  EXPECT_EQ(model->GetAllocatorType(), tvm::runtime::vm::AllocatorType::kNaive);

  delete model;
  EXPECT_EQ(SetEnv("DLR_RELAYVM_ALLOCATOR", ""), 0);
}

TEST(DLR, TestRelayVMAllocatorFromMetadata) {
  DLDevice dev = {static_cast<DLDeviceType>(kDLCPU), 0};
  std::string ro_file = "./ssd_mobilenet_v1/code.ro";
  std::string so_file = "./ssd_mobilenet_v1/compiled.so";
  std::string meta_file = "./ssd_mobilenet_v1/compiled.meta";
  std::ifstream ifs(meta_file);
  nlohmann::json metadata = nlohmann::json::parse(ifs);
  metadata["Model"]["RelayVMAllocator"] = "naive";
  std::string meta_str = metadata.dump();
  std::vector<DLRModelElem> model_elems = {
      {DLRModelElemType::RELAY_EXEC, ro_file.c_str(), nullptr, 0},
      {DLRModelElemType::TVM_LIB, so_file.c_str(), nullptr, 0},
      {DLRModelElemType::NEO_METADATA, nullptr, meta_str.c_str(), meta_str.size()}};
  dlr::RelayVMModel* model = new dlr::RelayVMModel(model_elems, dev);
  EXPECT_EQ(model->GetAllocatorType(), tvm::runtime::vm::AllocatorType::kNaive);
  delete model;
}
