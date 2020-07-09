#include "dlr_model.h"
#include "dlr_tvm.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "test_utils.hpp"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

class TVMTest : public ::testing::Test {
 protected:
  float* img;
  const int batch_size = 1;
  size_t img_size = 224 * 224 * 3;
  const int64_t input_shape[4] = {1, 224, 224, 3};
  const int input_dim = 4;
  const std::string model_path = "./resnet_v1_5_50";
  const std::string metadata_file = "./resnet_v1_5_50/compiled.meta";
  const std::string metadata_file_bak = "./resnet_v1_5_50/compiled.meta.bak";

  dlr::DLRModel* model;

  TVMTest() {
    // Setup input data
    img = LoadImageAndPreprocess("cat224-3.txt", img_size, batch_size);

    // Instantiate model
    int device_type = 1;
    int device_id = 0;
    std::vector<std::string> paths = {model_path};
    model = dlr::DLRModel::create_model(paths, device_type, device_id);
  }

  ~TVMTest() {
    delete model;
    delete img;
  }
};

TEST_F(TVMTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(TVMTest, TestGetInputName) {
  EXPECT_EQ(model->GetInputName(0), "input_tensor");
}

TEST_F(TVMTest, TestGetInputType) {
  EXPECT_EQ(model->GetInputType(0), "float32");
}

TEST_F(TVMTest, TestGetInput) {
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img, input_dim));
  float observed_input_data[img_size];
  EXPECT_NO_THROW(model->GetInput("input_tensor", observed_input_data));
  EXPECT_EQ(*img, observed_input_data[0]);
}

TEST_F(TVMTest, TestGetInputShape) {
  std::vector<int64_t> in_shape(std::begin(input_shape), std::end(input_shape));
  EXPECT_EQ(model->GetInputShape(0), in_shape);
}

TEST_F(TVMTest, TestGetInputSize) {
  EXPECT_EQ(model->GetInputSize(0), 1 * 224 * 224 * 3);
}

TEST_F(TVMTest, TestGetInputDim) { EXPECT_EQ(model->GetInputDim(0), 4); }

TEST_F(TVMTest, TestGetNumOutputs) { EXPECT_EQ(model->GetNumOutputs(), 2); }

TEST_F(TVMTest, TestGetOutputName) {
  EXPECT_EQ(model->GetOutputName(0), "ArgMax:0");
  EXPECT_EQ(model->GetOutputName(1), "softmax_tensor:0");
}

TEST_F(TVMTest, TestGetOutputType) {
  EXPECT_EQ(model->GetOutputType(0), "int32");
  EXPECT_EQ(model->GetOutputType(1), "float32");
}

TEST_F(TVMTest, TestGetOutputSize) { EXPECT_EQ(model->GetOutputSize(0), 1); }

TEST_F(TVMTest, TestGetOutputDim) { EXPECT_EQ(model->GetOutputSize(0), 1); }

TEST_F(TVMTest, TestRun) {
  void* outputs[2];
  outputs[0] = malloc(sizeof(float) * model->GetOutputSize(0));
  outputs[1] = malloc(sizeof(float) * model->GetOutputSize(1));
  EXPECT_NO_THROW(
      model->DLRModel::Run(1, reinterpret_cast<void**>(&img), outputs));
  float observed_input_data[img_size];
  EXPECT_NO_THROW(model->GetInput(0, observed_input_data));
  EXPECT_EQ(*img, observed_input_data[0]);
  free(outputs[0]);
  free(outputs[1]);
}

TEST_F(TVMTest, TestRunWithInputsAsDict) {
  std::map<std::string, void*> inputs;
  inputs.insert(std::make_pair("input_tensor", img));
  std::vector<void*> outputs(2);
  outputs[0] = malloc(sizeof(float) * model->GetOutputSize(0));
  outputs[1] = malloc(sizeof(float) * model->GetOutputSize(1));
  EXPECT_NO_THROW(
      model->DLRModel::Run(1, inputs, outputs));
  float observed_input_data[img_size];
  EXPECT_NO_THROW(model->GetInput(0, observed_input_data));
  EXPECT_EQ(*img, observed_input_data[0]);
  free(outputs[0]);
  free(outputs[1]);
}

TEST_F(TVMTest, TestTvmModelApisWithOutputMetadata) {
  EXPECT_EQ(model->HasMetadata(), true);
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img, input_dim));
  EXPECT_NO_THROW(model->Run());

  float output1[1];
  float output2[1];
  EXPECT_NO_THROW(model->GetOutput(0, output1));
  EXPECT_NO_THROW(
      model->GetOutputByName(model->GetOutputName(0).c_str(), output2));
  EXPECT_EQ(output1[0], output2[0]);

  try {
    model->GetOutputIndex("blah");
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Couldn't find index for output node blah!");
  }

  try {
    model->GetOutputByName("blah", output1);
  } catch (const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "Couldn't find index for output node blah!");
  }

  try {
    model->GetOutputName(2);
  } catch (const dmlc::Error& e) {
    std::string msg(e.what());
    EXPECT_TRUE(msg.find("Output index out of range."));
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
  DLContext ctx  = {
    static_cast<DLDeviceType>(device_type),
    device_id
  };
  std::vector<std::string> paths = {model_path};
  dlr::TVMModel model = dlr::TVMModel(paths, ctx);

  EXPECT_EQ(model.GetNumInputs(), 1);
  EXPECT_EQ(model.GetNumOutputs(), 2);
  EXPECT_EQ(model.HasMetadata(), false);
  try {
    model.GetOutputName(0);
  } catch(const dmlc::Error& e) {
    EXPECT_STREQ(e.what(), "No metadata file was found!");
  }
  try {
    model.GetOutputIndex("blah");
  } catch(const dmlc::Error& e) {
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
  mocked_metadata << "{}\n";
  mocked_metadata.close();

  const int device_type = 1;  // 1 - kDLCPU
  const int device_id = 0;
  DLContext ctx  = {
    static_cast<DLDeviceType>(device_type),
    device_id
  };
  std::vector<std::string> paths = {model_path};
  dlr::TVMModel model = dlr::TVMModel(paths, ctx);

  EXPECT_EQ(model.GetNumInputs(), 1);
  EXPECT_EQ(model.GetNumOutputs(), 2);
  EXPECT_EQ(model.HasMetadata(), true);

  try {
    model.GetOutputName(0);
  } catch(const dmlc::Error& e) {
    std::string msg(e.what());
    EXPECT_TRUE(msg.find("Output node with index 0 was not found in metadata file!"));
  }
  try {
    model.GetOutputIndex("blah");
  } catch(const dmlc::Error& e) {
    std::string msg(e.what());
    EXPECT_TRUE(msg.find("Output node with index 0 was not found in metadata file!"));
  }

  // put it back
  std::rename(metadata_file_bak.c_str(), metadata_file.c_str());
}
