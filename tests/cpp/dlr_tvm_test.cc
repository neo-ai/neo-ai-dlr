#include "dlr_tvm.h"

#include <gtest/gtest.h>

#include "dlr.h"
#include "test_utils.hpp"

class TVMTest : public ::testing::Test {
 protected:
  std::vector<float> img;
  const int batch_size = 1;
  size_t img_size = 224 * 224 * 3;
  const int64_t input_shape[4] = {1, 224, 224, 3};
  const int input_dim = 4;
  const std::string model_path = "./resnet_v1_5_50";
  const std::string metadata_file = "./resnet_v1_5_50/compiled.meta";
  const std::string metadata_file_bak = "./resnet_v1_5_50/compiled.meta.bak";

  dlr::TVMModel* model;

  TVMTest() {
    // Setup input data
    img = LoadImageAndPreprocess("cat224-3.txt", img_size, batch_size);

    // Instantiate model
    int device_type = 1;
    int device_id = 0;
    std::vector<std::string> paths = {model_path};
    DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
    model = new dlr::TVMModel(paths, ctx);
  }

  ~TVMTest() { delete model; }
};

TEST_F(TVMTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(TVMTest, TestGetInput) {
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  std::vector<float> observed_input_data(img_size);
  EXPECT_NO_THROW(model->GetInput("input_tensor", observed_input_data.data()));
  EXPECT_EQ(img, observed_input_data);
}

TEST_F(TVMTest, TestGetInputShape) {
  std::vector<int64_t> in_shape(std::begin(input_shape), std::end(input_shape));
  EXPECT_EQ(model->GetInputShape(0), in_shape);
}

TEST_F(TVMTest, TestGetInputSize) { EXPECT_EQ(model->GetInputSize(0), 1 * 224 * 224 * 3); }

TEST_F(TVMTest, TestGetInputDim) { EXPECT_EQ(model->GetInputDim(0), 4); }

TEST(TVM, TestTvmModelApisWithOutputMetadata) {
  const int device_type = 1;  // 1 - kDLCPU
  const int device_id = 0;
  DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
  std::vector<std::string> paths = {"./resnet_v1_5_50"};
  dlr::TVMModel model = dlr::TVMModel(paths, ctx);

  EXPECT_EQ(model.GetNumInputs(), 1);
  EXPECT_EQ(model.GetNumOutputs(), 2);
  EXPECT_EQ(model.HasMetadata(), true);
  EXPECT_STREQ(model.GetOutputName(0), "ArgMax:0");
  EXPECT_STREQ(model.GetOutputType(0), "int32");
  EXPECT_STREQ(model.GetOutputName(1), "softmax_tensor:0");
  EXPECT_STREQ(model.GetOutputType(1), "float32");

  const int batch_size = 1;
  size_t img_size = 224 * 224 * 3;
  std::vector<float> img = LoadImageAndPreprocess("cat224-3.txt", img_size, batch_size);
  img_size *= batch_size;
  int64_t shape[4] = {1, 224, 224, 3};

  int64_t output_size = 1;
  int output_dim;
  int index = 0;
  EXPECT_NO_THROW(model.GetOutputSizeDim(index, &output_size, &output_dim));

  EXPECT_NO_THROW(model.SetInput("input_tensor", shape, img.data(), 4));
  EXPECT_NO_THROW(model.Run());

  std::vector<float> output1(output_size);
  std::vector<float> output2(output_size);
  const float* output3;
  EXPECT_NO_THROW(model.GetOutput(index, output1.data()));
  EXPECT_NO_THROW(model.GetOutputByName(model.GetOutputName(index), output2.data()));
  EXPECT_NO_THROW(output3 = static_cast<const float*>(model.GetOutputPtr(index)));
  EXPECT_EQ(output1[0], output2[0]);
  EXPECT_EQ(output1[0], output3[0]);

  try {
    model.GetOutputIndex("blah");
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Couldn't find index for output node blah!");
  }

  try {
    model.GetOutputByName("blah", output1.data());
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Couldn't find index for output node blah!");
  }

  try {
    model.GetOutputName(2);
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Output node with index 2 was not found in metadata file!");
  }
}

TEST(TVM, TestTvmModelApisWithoutMetadata) {
  std::string model_path = "./resnet_v1_5_50";
  std::string metadata_file = model_path + "/compiled.meta";
  std::string metadata_file_bak = model_path + "/compiled.meta.bak";

  // rename metadata file such that metadata file is not found.
  std::rename(metadata_file.c_str(), metadata_file_bak.c_str());

  const int device_type = 1;  // 1 - kDLCPU
  const int device_id = 0;
  DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
  std::vector<std::string> paths = {model_path};
  dlr::TVMModel model = dlr::TVMModel(paths, ctx);

  EXPECT_EQ(model.GetNumInputs(), 1);
  EXPECT_EQ(model.GetNumOutputs(), 2);
  EXPECT_EQ(model.HasMetadata(), false);
  try {
    model.GetOutputName(0);
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "No metadata file was found!");
  }
  try {
    model.GetOutputIndex("blah");
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "No metadata file was found!");
  }

  // put it back
  std::rename(metadata_file_bak.c_str(), metadata_file.c_str());
}

TEST(TVM, TestTvmModelApisWithoutOutputInMetadata) {
  // rename metadata file such that metadata file is not found.
  std::string model_path = "./resnet_v1_5_50";
  std::string metadata_file = model_path + "/compiled.meta";
  std::string metadata_file_bak = model_path + "/compiled.meta.bak";

  std::rename(metadata_file.c_str(), metadata_file_bak.c_str());
  std::ofstream mocked_metadata;
  mocked_metadata.open(metadata_file.c_str());
  mocked_metadata << "{}";
  mocked_metadata.close();

  const int device_type = 1;  // 1 - kDLCPU
  const int device_id = 0;
  DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
  std::vector<std::string> paths = {model_path};
  dlr::TVMModel model = dlr::TVMModel(paths, ctx);

  EXPECT_EQ(model.GetNumInputs(), 1);
  EXPECT_EQ(model.GetNumOutputs(), 2);
  EXPECT_EQ(model.HasMetadata(), true);

  try {
    model.GetOutputName(0);
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Output node with index 0 was not found in metadata file!");
  }
  try {
    model.GetOutputIndex("blah");
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Output node with index 0 was not found in metadata file!");
  }

  // put it back
  std::rename(metadata_file_bak.c_str(), metadata_file.c_str());
}

TEST(TVM, TestTvmModelGetDeviceTypeFromMetadata) {
  // rename metadata file such that metadata file is not found.
  std::string model_path = "./resnet_v1_5_50";
  std::string metadata_file = model_path + "/compiled.meta";
  std::string metadata_file_bak = model_path + "/compiled.meta.bak";

  std::rename(metadata_file.c_str(), metadata_file_bak.c_str());
  std::ofstream mocked_metadata;
  mocked_metadata.open(metadata_file.c_str());
  mocked_metadata
      << "{\"Requirements\": { \"TargetDevice\": \"ML_C4\", \"TargetDeviceType\": \"cpu\"}}\n";
  mocked_metadata.close();

  DLContext ctx = {DLDeviceType::kDLCPU, 0};
  std::vector<std::string> paths = {model_path};
  dlr::TVMModel model = dlr::TVMModel(paths, ctx);

  EXPECT_EQ(model.GetNumInputs(), 1);
  EXPECT_EQ(model.GetNumOutputs(), 2);
  EXPECT_EQ(model.HasMetadata(), true);
  EXPECT_EQ(GetDLRDeviceType(model_path.data()), 1);
  EXPECT_EQ(model.GetDeviceTypeFromMetadata(), DLDeviceType::kDLCPU);

  mocked_metadata.open(metadata_file.c_str());
  mocked_metadata
      << "{\"Requirements\": { \"TargetDevice\": \"ML_C4\", \"TargetDeviceType\": \"gpu\"}}\n";
  mocked_metadata.close();

  EXPECT_THROW(({
                 try {
                   dlr::TVMModel(paths, ctx);
                 } catch (const dmlc::Error& e) {
                   EXPECT_STREQ(
                       e.what(),
                       "Compiled model requires device type \"gpu\" but user gave \"cpu\".");
                   throw;
                 }
               }),
               dmlc::Error);

  // put it back
  std::rename(metadata_file_bak.c_str(), metadata_file.c_str());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
