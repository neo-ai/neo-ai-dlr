#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include "dlr_tensorflow2/dlr_tensorflow2.h"

using namespace dlr;

TEST(PrepareTFConfigProto1, test1) {
  DLR_TF2Config tf2_config = {};
  tf2_config.intra_op_parallelism_threads = 2;
  tf2_config.inter_op_parallelism_threads = 3;
  tf2_config.gpu_options.per_process_gpu_memory_fraction = 0.33;
  tf2_config.gpu_options.allow_growth = 1;
  std::vector<std::uint8_t> config;
  PrepareTF2ConfigProto(tf2_config, config);
  std::uint8_t exp[] = {0x10, 0x2,  0x28, 0x3,  0x32, 0xb,  0x9,  0x1f, 0x85,
                        0xeb, 0x51, 0xb8, 0x1e, 0xd5, 0x3f, 0x20, 0x1};
  EXPECT_TRUE(std::equal(std::begin(exp), std::end(exp), config.begin()));
}

TEST(PrepareTFConfigProto1, test2) {
  DLR_TF2Config tf2_config = {};
  tf2_config.intra_op_parallelism_threads = 2;
  tf2_config.inter_op_parallelism_threads = 3;
  tf2_config.gpu_options.allow_growth = 1;
  std::vector<std::uint8_t> config;
  PrepareTF2ConfigProto(tf2_config, config);
  std::uint8_t exp[] = {0x10, 0x2, 0x28, 0x3, 0x32, 0x2, 0x20, 0x1};
  EXPECT_TRUE(std::equal(std::begin(exp), std::end(exp), config.begin()));
}

TEST(PrepareTFConfigProto1, test3) {
  DLR_TF2Config tf2_config = {};
  tf2_config.intra_op_parallelism_threads = 2;
  tf2_config.inter_op_parallelism_threads = 3;
  std::vector<std::uint8_t> config;
  PrepareTF2ConfigProto(tf2_config, config);
  std::uint8_t exp[] = {0x10, 0x2, 0x28, 0x3};
  EXPECT_TRUE(std::equal(std::begin(exp), std::end(exp), config.begin()));
}

TEST(PrepareTFConfigProto1, test4) {
  DLR_TF2Config tf2_config = {};
  std::vector<std::uint8_t> config;
  PrepareTF2ConfigProto(tf2_config, config);
  EXPECT_TRUE(config.empty());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
