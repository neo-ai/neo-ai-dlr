#include <gtest/gtest.h>

#include "dlr.h"
#include "dlr_tvm.h"
#include "test_utils.hpp"

class TVMElemTest : public ::testing::Test {
 protected:
  std::vector<float> img;
  const int batch_size = 1;
  size_t img_size = 224 * 224 * 3;
  const int64_t input_shape[4] = {1, 224, 224, 3};
  const int input_dim = 4;
  const std::string graph_file = "./resnet_v1_5_50/compiled_model.json";
  const std::string params_file = "./resnet_v1_5_50/compiled.params";
  const std::string so_file = "./resnet_v1_5_50/compiled.so";
  const std::string meta_file = "./resnet_v1_5_50/compiled.meta";

  dlr::TVMModel* model;

  TVMElemTest() {
    std::string graph_str = dlr::LoadFileToString(graph_file);
    std::string params_str = dlr::LoadFileToString(params_file, std::ios::in | std::ios::binary);
    std::string meta_str = dlr::LoadFileToString(meta_file);

    std::vector<DLRModelElem> model_elems = {
        {DLRModelElemType::TVM_GRAPH, nullptr, graph_str.c_str(), 0},
        {DLRModelElemType::TVM_PARAMS, nullptr, params_str.data(), params_str.size()},
        {DLRModelElemType::TVM_LIB, so_file.c_str(), nullptr, 0},
        {DLRModelElemType::NEO_METADATA, nullptr, meta_str.c_str(), 0}};

    // Setup input data
    img = LoadImageAndPreprocess("cat224-3.txt", img_size, batch_size);

    // Instantiate model
    int device_type = 1;
    int device_id = 0;
    DLContext ctx = {static_cast<DLDeviceType>(device_type), device_id};
    model = new dlr::TVMModel(model_elems, ctx);
  }

  ~TVMElemTest() { delete model; }
};

TEST_F(TVMElemTest, TestGetNumInputs) { EXPECT_EQ(model->GetNumInputs(), 1); }

TEST_F(TVMElemTest, TestGetInput) {
  EXPECT_NO_THROW(model->SetInput("input_tensor", input_shape, img.data(), input_dim));
  std::vector<float> observed_input_data(img_size);
  EXPECT_NO_THROW(model->GetInput("input_tensor", observed_input_data.data()));
  EXPECT_EQ(img, observed_input_data);
}

TEST_F(TVMElemTest, TestGetInputShape) {
  std::vector<int64_t> in_shape(std::begin(input_shape), std::end(input_shape));
  EXPECT_EQ(model->GetInputShape(0), in_shape);
}

TEST_F(TVMElemTest, TestGetInputSize) { EXPECT_EQ(model->GetInputSize(0), 1 * 224 * 224 * 3); }

TEST_F(TVMElemTest, TestGetInputDim) { EXPECT_EQ(model->GetInputDim(0), 4); }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
