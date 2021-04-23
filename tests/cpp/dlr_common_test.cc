#include "dlr_common.h"

#include <gtest/gtest.h>

#include "dlr.h"

TEST(DLRCommon, GetBasename) {
  EXPECT_EQ(dlr::GetBasename("/usr/lib"), "lib");
  EXPECT_EQ(dlr::GetBasename("/usr/lib/"), "lib");
  EXPECT_EQ(dlr::GetBasename("usr"), "usr");
  EXPECT_EQ(dlr::GetBasename("/"), "/");
  EXPECT_EQ(dlr::GetBasename("."), ".");
  EXPECT_EQ(dlr::GetBasename(".."), "..");
  // Windows
  EXPECT_EQ(dlr::GetBasename("C:\\Windows\\Libraries"), "Libraries");
  EXPECT_EQ(dlr::GetBasename("C:\\Windows\\Libraries\\"), "Libraries");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}
