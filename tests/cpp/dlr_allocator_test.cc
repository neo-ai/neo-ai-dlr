#include "dlr_allocator.h"
#include "dlr.h"
#include "dlr_tvm.h"

#include <gtest/gtest.h>

#include "test_utils.hpp"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifndef _WIN32
  testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif  // _WIN32
  return RUN_ALL_TESTS();
}

class CustomAllocatorTest : public ::testing::Test {
 protected:
  CustomAllocatorTest() { dlr::DLRAllocatorFunctions::Clear(); }
};

void* test_malloc(size_t size) { throw dmlc::Error("Using custom alloc"); }

void test_free(void* ptr) { throw dmlc::Error("Using custom free"); }

void* test_memalign(size_t alignment, size_t size) { throw dmlc::Error("Using custom memalign"); }

TEST_F(CustomAllocatorTest, CustomAllocatorsUnused) {
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AllSet());
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AnySet());
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMallocFunction(), nullptr);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetFreeFunction(), nullptr);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMemalignFunction(), nullptr);
  void* p = nullptr;
  EXPECT_NO_THROW(p = dlr::DLRAllocatorFunctions::Malloc(1));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::Free(p));
}

TEST_F(CustomAllocatorTest, CustomAllocatorsOnlyMalloc) {
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AllSet());
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AnySet());
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMallocFunction(), nullptr);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetFreeFunction(), nullptr);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMemalignFunction(), nullptr);
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMallocFunction(test_malloc));
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AllSet());
  EXPECT_TRUE(dlr::DLRAllocatorFunctions::AnySet());
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMallocFunction(), test_malloc);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetFreeFunction(), nullptr);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMemalignFunction(), nullptr);
}

TEST_F(CustomAllocatorTest, CustomAllocatorsUsed) {
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AllSet());
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMallocFunction(test_malloc));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetFreeFunction(test_free));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMemalignFunction(test_memalign));
  EXPECT_TRUE(dlr::DLRAllocatorFunctions::AllSet());
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMallocFunction(), test_malloc);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetFreeFunction(), test_free);
  EXPECT_EQ(dlr::DLRAllocatorFunctions::GetMemalignFunction(), test_memalign);
  EXPECT_THROW(
      {
        try {
          dlr::DLRAllocatorFunctions::Malloc(4);
        } catch (const dmlc::Error& e) {
          EXPECT_STREQ(e.what(), "Using custom alloc");
          throw;
        }
      },
      dmlc::Error);
  EXPECT_THROW(
      {
        try {
          dlr::DLRAllocatorFunctions::Free(nullptr);
        } catch (const dmlc::Error& e) {
          EXPECT_STREQ(e.what(), "Using custom free");
          throw;
        }
      },
      dmlc::Error);
}

TEST_F(CustomAllocatorTest, CustomAllocatorsSTLUnset) {
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AllSet());
  std::vector<int, dlr::DLRAllocator<int>>* data;
  EXPECT_NO_THROW((data = new std::vector<int, dlr::DLRAllocator<int>>(1024)));
  EXPECT_NO_THROW((delete data));
}

TEST_F(CustomAllocatorTest, CustomAllocatorsSTLSet) {
  EXPECT_FALSE(dlr::DLRAllocatorFunctions::AllSet());
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMallocFunction(test_malloc));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetFreeFunction(test_free));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMemalignFunction(test_memalign));
  EXPECT_TRUE(dlr::DLRAllocatorFunctions::AllSet());

  EXPECT_THROW(({
                 std::vector<int, dlr::DLRAllocator<int>>* data;
                 try {
                   data = new std::vector<int, dlr::DLRAllocator<int>>(1024);
                 } catch (const dmlc::Error& e) {
                   EXPECT_STREQ(e.what(), "Using custom alloc");
                   throw;
                 }
               }),
               dmlc::Error);

  EXPECT_THROW(({
                 std::vector<int, dlr::DLRAllocator<int>>* data =
                     new std::vector<int, dlr::DLRAllocator<int>>(256);
                 try {
                   delete data;
                 } catch (const dmlc::Error& e) {
                   EXPECT_STREQ(e.what(), "Using custom free");
                   throw;
                 }
               }),
               dmlc::Error);
}

class CustomAllocatorTrackingTest : public CustomAllocatorTest {
 protected:
  ~CustomAllocatorTrackingTest() {
    malloc_calls_.clear();
    free_calls_.clear();
    memalign_calls_.clear();
  }

  size_t TotalBytesAllocated() {
    size_t sum = 0;
    for (auto& pair : malloc_calls_) {
      sum += pair.first;
    }
    for (auto& tuple : memalign_calls_) {
      sum += std::get<1>(tuple);
    }
    return sum;
  }

 public:
  static std::vector<std::pair<size_t, void*>> malloc_calls_;
  static std::vector<void*> free_calls_;
  static std::vector<std::tuple<size_t, size_t, void*>> memalign_calls_;
};

std::vector<std::pair<size_t, void*>> CustomAllocatorTrackingTest::malloc_calls_;
std::vector<void*> CustomAllocatorTrackingTest::free_calls_;
std::vector<std::tuple<size_t, size_t, void*>> CustomAllocatorTrackingTest::memalign_calls_;

void* tracking_malloc(size_t size) {
  void* ptr = malloc(size);
  CustomAllocatorTrackingTest::malloc_calls_.push_back({size, ptr});
  return ptr;
}

void tracking_free(void* ptr) {
  CustomAllocatorTrackingTest::free_calls_.push_back(ptr);
  free(ptr);
}

void* tracking_memalign(size_t alignment, size_t size) {
  void* ptr;
#if _MSC_VER
  ptr = _aligned_malloc(size, alignment);
  if (ptr == nullptr) throw std::bad_alloc();
#else
  // posix_memalign is available in android ndk since __ANDROID_API__ >= 17
  int ret = posix_memalign(&ptr, alignment, size);
  if (ret != 0) throw std::bad_alloc();
#endif
  CustomAllocatorTrackingTest::memalign_calls_.push_back({alignment, size, ptr});
  return ptr;
}

TEST_F(CustomAllocatorTrackingTest, CustomAllocatorsTvm) {
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMallocFunction(tracking_malloc));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetFreeFunction(tracking_free));
  EXPECT_NO_THROW(dlr::DLRAllocatorFunctions::SetMemalignFunction(tracking_memalign));
  DLRModelHandle model = nullptr;
  EXPECT_EQ(CustomAllocatorTrackingTest::malloc_calls_.size(), 0);
  EXPECT_EQ(CustomAllocatorTrackingTest::free_calls_.size(), 0);
  EXPECT_EQ(CustomAllocatorTrackingTest::memalign_calls_.size(), 0);
  EXPECT_EQ(TotalBytesAllocated(), 0);

  // Test that memalign, free, malloc have been called after loading the model.
  EXPECT_EQ(CreateDLRModel(&model, "./resnet_v1_5_50", /*device_type=*/1, 0), 0);
  EXPECT_GT(CustomAllocatorTrackingTest::malloc_calls_.size(), 0);
  EXPECT_GT(CustomAllocatorTrackingTest::free_calls_.size(), 0);
  EXPECT_GT(CustomAllocatorTrackingTest::memalign_calls_.size(), 0);
  // Giving some tolerance for platform differences.
  EXPECT_NEAR(TotalBytesAllocated(), 585280062, 100000);

  size_t img_size = 224 * 224 * 3;
  std::vector<float> img = LoadImageAndPreprocess("cat224-3.txt", img_size, 1);
  int64_t shape[4] = {1, 224, 224, 3};
  const char* input_name = "input_tensor";
  EXPECT_EQ(SetDLRInput(&model, input_name, shape, img.data(), 4), 0);
  EXPECT_EQ(RunDLRModel(&model), 0);
  // Input/output buffer was allocated.
  EXPECT_NEAR(TotalBytesAllocated(), 586181182, 100000);

  // Check output is correct.
  int output[1];
  EXPECT_EQ(GetDLROutput(&model, 0, output), 0);
  EXPECT_EQ(output[0], 112);

  // Free should be called.
  size_t free_count_before = CustomAllocatorTrackingTest::free_calls_.size();
  EXPECT_EQ(DeleteDLRModel(&model), 0);
  size_t free_count_after = CustomAllocatorTrackingTest::free_calls_.size();
  EXPECT_GT(free_count_after, free_count_before);
}
