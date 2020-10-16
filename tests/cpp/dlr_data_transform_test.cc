#include "dlr_relayvm.h"

#include <gtest/gtest.h>
#include "test_utils.hpp"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

TEST(DLR, RelayVMDataTransform) {
  DLContext ctx = {kDLCPU, 0};
  std::vector<std::string> paths = {"./onehotencoder"};
  dlr::RelayVMModel* model = new dlr::RelayVMModel(paths, ctx);

  const char* data = "[[\"apple\", 1, 7], [\"banana\", 3, 8], [\"squash\", 2, 9]]";
  std::vector<int64_t> shape = {static_cast<int64_t>(std::strlen(data))};
  model->SetInput("input", shape.data(), const_cast<char*>(data), 1);
  model->Run();

  int64_t size;
  int dim;
  EXPECT_NO_THROW(model->GetOutputSizeDim(0, &size, &dim));
  EXPECT_EQ(size, 3 * 8);
  EXPECT_EQ(dim, 2);
  int64_t output_shape[2];
  EXPECT_NO_THROW(model->GetOutputShape(0, output_shape));
  EXPECT_EQ(output_shape[0], 3);
  EXPECT_EQ(output_shape[1], 8);

  std::vector<float> expected_output = {1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                        1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> output(size, 0);
  EXPECT_NO_THROW(model->GetOutput(0, output.data()));
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(output[i], expected_output[i]) << "Output at index " << i;
  }
  delete model;
}
