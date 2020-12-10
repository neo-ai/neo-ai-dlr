#include <dlfcn.h>
#include <gtest/gtest.h>
#include <stdint.h>

#include <fstream>

#include "../test_utils.hpp"

typedef void* DLRModelHandle;
struct DLRFunctions {
  void* dlr;
  int (*CreateDLRModel)(DLRModelHandle* handle, const char* model_path, int dev_type, int dev_id);
  int (*DeleteDLRModel)(DLRModelHandle* handle);
  int (*RunDLRModel)(DLRModelHandle* handle);
  int (*GetDLRNumInputs)(DLRModelHandle* handle, int* num_inputs);
  int (*GetDLRNumWeights)(DLRModelHandle* handle, int* num_weights);
  int (*GetDLRInputName)(DLRModelHandle* handle, int index, const char** input_name);
  int (*GetDLRInputType)(DLRModelHandle* handle, int index, const char** input_type);
  int (*GetDLRWeightName)(DLRModelHandle* handle, int index, const char** weight_name);
  int (*SetDLRInput)(DLRModelHandle* handle, const char* name, const int64_t* shape,
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
};

DLRFunctions InitDLR(const std::string& lib_path) {
  DLRFunctions f;
  void* dlr = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  EXPECT_NE(dlr, nullptr) << dlerror();
  f.dlr = dlr;
  *(void**)(&f.CreateDLRModel) = dlsym(dlr, "CreateDLRModel");
  *(void**)(&f.DeleteDLRModel) = dlsym(dlr, "DeleteDLRModel");
  *(void**)(&f.RunDLRModel) = dlsym(dlr, "RunDLRModel");
  *(void**)(&f.GetDLRNumInputs) = dlsym(dlr, "GetDLRNumInputs");
  *(void**)(&f.GetDLRNumWeights) = dlsym(dlr, "GetDLRNumWeights");
  *(void**)(&f.GetDLRInputName) = dlsym(dlr, "GetDLRInputName");
  *(void**)(&f.GetDLRInputType) = dlsym(dlr, "GetDLRInputType");
  *(void**)(&f.GetDLRWeightName) = dlsym(dlr, "GetDLRWeightName");
  *(void**)(&f.SetDLRInput) = dlsym(dlr, "SetDLRInput");
  *(void**)(&f.GetDLRInput) = dlsym(dlr, "GetDLRInput");
  *(void**)(&f.GetDLRInputShape) = dlsym(dlr, "GetDLRInputShape");
  *(void**)(&f.GetDLRInputSizeDim) = dlsym(dlr, "GetDLRInputSizeDim");
  *(void**)(&f.GetDLROutputShape) = dlsym(dlr, "GetDLROutputShape");
  *(void**)(&f.GetDLROutput) = dlsym(dlr, "GetDLROutput");
  *(void**)(&f.GetDLROutputPtr) = dlsym(dlr, "GetDLROutputPtr");
  *(void**)(&f.GetDLRNumOutputs) = dlsym(dlr, "GetDLRNumOutputs");
  *(void**)(&f.GetDLROutputSizeDim) = dlsym(dlr, "GetDLROutputSizeDim");
  *(void**)(&f.GetDLROutputType) = dlsym(dlr, "GetDLROutputType");
  *(void**)(&f.GetDLRHasMetadata) = dlsym(dlr, "GetDLRHasMetadata");
  *(void**)(&f.GetDLROutputName) = dlsym(dlr, "GetDLROutputName");
  *(void**)(&f.GetDLROutputIndex) = dlsym(dlr, "GetDLROutputIndex");
  *(void**)(&f.GetDLROutputByName) = dlsym(dlr, "GetDLROutputByName");
  *(void**)(&f.DLRGetLastError) = dlsym(dlr, "DLRGetLastError");
  *(void**)(&f.GetDLRBackend) = dlsym(dlr, "GetDLRBackend");
  *(void**)(&f.GetDLRDeviceType) = dlsym(dlr, "GetDLRDeviceType");
  return f;
}

DLRModelHandle GetDLRModel(const DLRFunctions& dlr) {
  DLRModelHandle model = nullptr;
  const char* model_path = "./resnet_v1_5_50";
  int device_type = 1;  // cpu;
  if (dlr.CreateDLRModel(&model, model_path, device_type, 0) != 0) {
    LOG(INFO) << dlr.DLRGetLastError() << std::endl;
    throw std::runtime_error("Could not load DLR Model");
  }
  return model;
}

void CopyFile(const std::string& src, const std::string& dst) {
  std::ifstream source(src, std::ios::binary);
  std::ofstream dest(dst, std::ios::binary);
  dest << source.rdbuf();
}

void RunInference(const DLRFunctions& dlr, DLRModelHandle model, const std::vector<float>& image) {
  int64_t shape[4] = {1, 224, 224, 3};
  const char* input_name = "input_tensor";
  EXPECT_EQ(dlr.SetDLRInput(&model, input_name, shape, image.data(), 4), 0);
  EXPECT_EQ(dlr.RunDLRModel(&model), 0);
  // output 0
  int output0[1];
  EXPECT_EQ(dlr.GetDLROutput(&model, 0, output0), 0);
  EXPECT_EQ(output0[0], 112);
  EXPECT_EQ(dlr.GetDLROutputByName(&model, "ArgMax:0", output0), 0);
  EXPECT_EQ(output0[0], 112);
  const void* output0_p;
  EXPECT_EQ(dlr.GetDLROutputPtr(&model, 0, &output0_p), 0);
  EXPECT_EQ(((int*)output0_p)[0], 112);
  // output 1
  float output1[1001];
  EXPECT_EQ(dlr.GetDLROutput(&model, 1, output1), 0);
  EXPECT_GT(output1[112], 0.01);
  EXPECT_EQ(dlr.GetDLROutputByName(&model, "softmax_tensor:0", output1), 0);
  EXPECT_GT(output1[112], 0.01);
  float* output1_p;
  EXPECT_EQ(dlr.GetDLROutputPtr(&model, 1, (const void**)&output1_p), 0);
  EXPECT_GT(output1_p[112], 0.01);
  for (int i = 0; i < 1001; i++) {
    EXPECT_EQ(output1_p[i], output1[i]);
  }
}

TEST(DLR, TestRunDLRModel_MultipleLibs) {
  // dlopen() will not load libdlr.so twice if the path is the same, so make a copy of it with a new
  // name.
  std::string dlr_copy_path = testing::TempDir() + "/libdlr2.so";
  CopyFile("lib/libdlr.so", dlr_copy_path);
  DLRFunctions dlr_0 = InitDLR("lib/libdlr.so");
  DLRFunctions dlr_1 = InitDLR(dlr_copy_path);

  size_t img_size = 224 * 224 * 3;
  std::vector<float> img = LoadImageAndPreprocess("cat224-3.txt", img_size, 1);

  auto model_0 = GetDLRModel(dlr_0);
  auto model_1 = GetDLRModel(dlr_1);
  RunInference(dlr_0, model_0, img);
  RunInference(dlr_1, model_1, img);

  dlr_0.DeleteDLRModel(&model_0);
  dlr_1.DeleteDLRModel(&model_1);
  dlclose(dlr_0.dlr);
  dlclose(dlr_1.dlr);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
