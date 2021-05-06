#include <dlfcn.h>
#include <gtest/gtest.h>
#include <stdint.h>

#include "../test_utils.hpp"

typedef void* DLRModelHandle;
int (*CreateDLRModel)(DLRModelHandle* handle, const char* model_path, int dev_type, int dev_id);
int (*DeleteDLRModel)(DLRModelHandle* handle);
int (*RunDLRModel)(DLRModelHandle* handle);
int (*GetDLRNumInputs)(DLRModelHandle* handle, int* num_inputs);
int (*GetDLRNumWeights)(DLRModelHandle* handle, int* num_weights);
int (*GetDLRInputName)(DLRModelHandle* handle, int index, const char** input_name);
int (*GetDLRInputType)(DLRModelHandle* handle, int index, const char** input_type);
int (*GetDLRWeightName)(DLRModelHandle* handle, int index, const char** weight_name);
int (*SetDLRInput)(DLRModelHandle* handle, const char* name, const int64_t* shape, void* input,
                   int dim);
// New signature of SetDLRInput - input data pointer marked as const
int (*SetDLRInputC)(DLRModelHandle* handle, const char* name, const int64_t* shape,
                    const void* input, int dim);
int (*GetDLRInput)(DLRModelHandle* handle, const char* name, void* input);
int (*GetDLRInputShape)(DLRModelHandle* handle, int index, int64_t* shape);
int (*GetDLRInputSizeDim)(DLRModelHandle* handle, int index, int64_t* size, int* dim);
int (*GetDLROutputShape)(DLRModelHandle* handle, int index, int64_t* shape);
int (*GetDLROutput)(DLRModelHandle* handle, int index, void* out);
int (*GetDLROutputPtr)(DLRModelHandle* handle, int index, const void** out);
int (*GetDLRNumOutputs)(DLRModelHandle* handle, int* num_outputs);
int (*GetDLROutputSizeDim)(DLRModelHandle* handle, int index, int64_t* size, int* dim);
int (*GetDLROutputType)(DLRModelHandle* handle, int index, const char** output_type);
int (*GetDLRHasMetadata)(DLRModelHandle* handle, bool* has_metadata);
int (*GetDLROutputName)(DLRModelHandle* handle, const int index, const char** name);
int (*GetDLROutputIndex)(DLRModelHandle* handle, const char* name, int* index);
int (*GetDLROutputByName)(DLRModelHandle* handle, const char* name, void* out);
const char* (*DLRGetLastError)();
int (*GetDLRBackend)(DLRModelHandle* handle, const char** name);
int (*GetDLRDeviceType)(const char* model_path);

void InitDLR() {
  void* dlr = dlopen("lib/libdlr.so", RTLD_LAZY | RTLD_LOCAL);
  *(void**)(&CreateDLRModel) = dlsym(dlr, "CreateDLRModel");
  *(void**)(&DeleteDLRModel) = dlsym(dlr, "DeleteDLRModel");
  *(void**)(&RunDLRModel) = dlsym(dlr, "RunDLRModel");
  *(void**)(&GetDLRNumInputs) = dlsym(dlr, "GetDLRNumInputs");
  *(void**)(&GetDLRNumWeights) = dlsym(dlr, "GetDLRNumWeights");
  *(void**)(&GetDLRInputName) = dlsym(dlr, "GetDLRInputName");
  *(void**)(&GetDLRInputType) = dlsym(dlr, "GetDLRInputType");
  *(void**)(&GetDLRWeightName) = dlsym(dlr, "GetDLRWeightName");
  *(void**)(&SetDLRInput) = dlsym(dlr, "SetDLRInput");
  *(void**)(&SetDLRInputC) = dlsym(dlr, "SetDLRInput");
  *(void**)(&GetDLRInput) = dlsym(dlr, "GetDLRInput");
  *(void**)(&GetDLRInputShape) = dlsym(dlr, "GetDLRInputShape");
  *(void**)(&GetDLRInputSizeDim) = dlsym(dlr, "GetDLRInputSizeDim");
  *(void**)(&GetDLROutputShape) = dlsym(dlr, "GetDLROutputShape");
  *(void**)(&GetDLROutput) = dlsym(dlr, "GetDLROutput");
  *(void**)(&GetDLROutputPtr) = dlsym(dlr, "GetDLROutputPtr");
  *(void**)(&GetDLRNumOutputs) = dlsym(dlr, "GetDLRNumOutputs");
  *(void**)(&GetDLROutputSizeDim) = dlsym(dlr, "GetDLROutputSizeDim");
  *(void**)(&GetDLROutputType) = dlsym(dlr, "GetDLROutputType");
  *(void**)(&GetDLRHasMetadata) = dlsym(dlr, "GetDLRHasMetadata");
  *(void**)(&GetDLROutputName) = dlsym(dlr, "GetDLROutputName");
  *(void**)(&GetDLROutputIndex) = dlsym(dlr, "GetDLROutputIndex");
  *(void**)(&GetDLROutputByName) = dlsym(dlr, "GetDLROutputByName");
  *(void**)(&DLRGetLastError) = dlsym(dlr, "DLRGetLastError");
  *(void**)(&GetDLRBackend) = dlsym(dlr, "GetDLRBackend");
  *(void**)(&GetDLRDeviceType) = dlsym(dlr, "GetDLRDeviceType");
}

DLRModelHandle GetDLRModel() {
  DLRModelHandle model = nullptr;
  const char* model_path = "./resnet_v1_5_50";
  int device_type = 1;  // cpu;
  if (CreateDLRModel(&model, model_path, device_type, 0) != 0) {
    LOG(INFO) << DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  return model;
}

TEST(DLR, TestGetDLRDeviceType) {
  const char* model_path = "./resnet_v1_5_50";
  EXPECT_EQ(GetDLRDeviceType(model_path), -1);
}

TEST(DLR, TestGetDLRNumInputs) {
  auto model = GetDLRModel();
  int num_inputs;
  EXPECT_EQ(GetDLRNumInputs(&model, &num_inputs), 0);
  EXPECT_EQ(num_inputs, 1);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRNumWeights) {
  auto model = GetDLRModel();
  int num_weights;
  EXPECT_EQ(GetDLRNumWeights(&model, &num_weights), 0);
  EXPECT_EQ(num_weights, 108);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRInputName) {
  auto model = GetDLRModel();
  const char* input_name;
  EXPECT_EQ(GetDLRInputName(&model, 0, &input_name), 0);
  EXPECT_STREQ(input_name, "input_tensor");
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRInputType) {
  auto model = GetDLRModel();
  const char* input_type;
  EXPECT_EQ(GetDLRInputType(&model, 0, &input_type), 0);
  EXPECT_STREQ(input_type, "float32");
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRWeightName) {
  auto model = GetDLRModel();
  const char* weight_name;
  EXPECT_EQ(GetDLRWeightName(&model, 0, &weight_name), 0);
  EXPECT_STREQ(weight_name, "p45");
  EXPECT_EQ(GetDLRWeightName(&model, 107, &weight_name), 0);
  EXPECT_STREQ(weight_name, "p72");
  DeleteDLRModel(&model);
}

TEST(DLR, TestSetDLRInput) {
  auto model = GetDLRModel();
  size_t img_size = 224 * 224 * 3;
  std::vector<float> img = LoadImageAndPreprocess("cat224-3.txt", img_size, 1);
  int64_t shape[4] = {1, 224, 224, 3};
  const char* input_name = "input_tensor";
  std::vector<float> img2(img_size);
  float* img_data = img.data();
  const float* cimg_data = img.data();
  // Test that old SetDLRInput (with non-const input) still works
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, img_data, 4), 0);
  // Test that new SetDLRInput accepts const and non-const inputs
  EXPECT_EQ(SetDLRInputC(&model, input_name, shape, img_data, 4), 0);
  EXPECT_EQ(SetDLRInputC(&model, input_name, shape, cimg_data, 4), 0);
  EXPECT_EQ(GetDLRInput(&model, input_name, img2.data()), 0);
  EXPECT_EQ(img, img2);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRInputShape) {
  auto model = GetDLRModel();
  int64_t input_size;
  int input_dim;
  int index = 0;
  EXPECT_EQ(GetDLRInputSizeDim(&model, 0, &input_size, &input_dim), 0);
  EXPECT_EQ(input_dim, 4);
  EXPECT_EQ(input_size, 1 * 224 * 224 * 3);
  std::vector<int64_t> shape(input_dim);
  EXPECT_EQ(GetDLRInputShape(&model, 0, shape.data()), 0);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 224);
  EXPECT_EQ(shape[2], 224);
  EXPECT_EQ(shape[3], 3);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputShape) {
  auto model = GetDLRModel();
  int64_t output_size;
  int output_dim;
  int index = 0;
  // output 0
  EXPECT_EQ(GetDLROutputSizeDim(&model, 0, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 1);
  EXPECT_EQ(output_size, 1);
  std::vector<int64_t> shape0(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 0, shape0.data()), 0);
  EXPECT_EQ(shape0[0], 1);
  // output 1
  EXPECT_EQ(GetDLROutputSizeDim(&model, 1, &output_size, &output_dim), 0);
  EXPECT_EQ(output_dim, 2);
  EXPECT_EQ(output_size, 1001);
  std::vector<int64_t> shape1(output_dim);
  EXPECT_EQ(GetDLROutputShape(&model, 1, shape1.data()), 0);
  EXPECT_EQ(shape1[0], 1);
  EXPECT_EQ(shape1[1], 1001);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRNumOutputs) {
  auto model = GetDLRModel();
  int num_outputs;
  EXPECT_EQ(GetDLRNumOutputs(&model, &num_outputs), 0);
  EXPECT_EQ(num_outputs, 2);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputType) {
  auto model = GetDLRModel();
  // output 0
  const char* output_type0;
  EXPECT_EQ(GetDLROutputType(&model, 0, &output_type0), 0);
  EXPECT_STREQ(output_type0, "int32");
  // output 1
  const char* output_type1;
  EXPECT_EQ(GetDLROutputType(&model, 1, &output_type1), 0);
  EXPECT_STREQ(output_type1, "float32");
  DeleteDLRModel(&model);
}

TEST(DLR, GetDLRHasMetadata) {
  auto model = GetDLRModel();
  bool has_metadata;
  EXPECT_EQ(GetDLRHasMetadata(&model, &has_metadata), 0);
  EXPECT_TRUE(has_metadata);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputName) {
  auto model = GetDLRModel();
  // output 0
  const char* output_name0;
  EXPECT_EQ(GetDLROutputName(&model, 0, &output_name0), 0);
  EXPECT_STREQ(output_name0, "ArgMax:0");
  // output 1
  const char* output_name1;
  EXPECT_EQ(GetDLROutputName(&model, 1, &output_name1), 0);
  EXPECT_STREQ(output_name1, "softmax_tensor:0");
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLROutputIndex) {
  auto model = GetDLRModel();
  // output 0
  int output_index0;
  EXPECT_EQ(GetDLROutputIndex(&model, "ArgMax:0", &output_index0), 0);
  EXPECT_EQ(output_index0, 0);
  // output 1
  int output_index1;
  EXPECT_EQ(GetDLROutputIndex(&model, "softmax_tensor:0", &output_index1), 0);
  EXPECT_EQ(output_index1, 1);
  DeleteDLRModel(&model);
}

TEST(DLR, TestGetDLRBackend) {
  auto model = GetDLRModel();
  const char* backend;
  EXPECT_EQ(GetDLRBackend(&model, &backend), 0);
  EXPECT_STREQ(backend, "tvm");
  DeleteDLRModel(&model);
}

TEST(DLR, TestRunDLRModel_GetDLROutput) {
  auto model = GetDLRModel();
  size_t img_size = 224 * 224 * 3;
  std::vector<float> img = LoadImageAndPreprocess("cat224-3.txt", img_size, 1);
  int64_t shape[4] = {1, 224, 224, 3};
  const char* input_name = "input_tensor";
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, img.data(), 4), 0);
  EXPECT_EQ(RunDLRModel(&model), 0);
  // output 0
  int output0[1];
  EXPECT_EQ(GetDLROutput(&model, 0, output0), 0);
  EXPECT_EQ(output0[0], 112);
  EXPECT_EQ(GetDLROutputByName(&model, "ArgMax:0", output0), 0);
  EXPECT_EQ(output0[0], 112);
  const void* output0_p;
  EXPECT_EQ(GetDLROutputPtr(&model, 0, &output0_p), 0);
  EXPECT_EQ(((int*)output0_p)[0], 112);
  // output 1
  float output1[1001];
  EXPECT_EQ(GetDLROutput(&model, 1, output1), 0);
  EXPECT_GT(output1[112], 0.01);
  EXPECT_EQ(GetDLROutputByName(&model, "softmax_tensor:0", output1), 0);
  EXPECT_GT(output1[112], 0.01);
  float* output1_p;
  EXPECT_EQ(GetDLROutputPtr(&model, 1, (const void**)&output1_p), 0);
  EXPECT_GT(output1_p[112], 0.01);
  for (int i = 0; i < 1001; i++) {
    EXPECT_EQ(output1_p[i], output1[i]);
  }
  DeleteDLRModel(&model);
}

int main(int argc, char** argv) {
  InitDLR();
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
