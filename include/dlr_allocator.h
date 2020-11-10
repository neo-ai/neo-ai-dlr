#ifndef DLR_ALLOCATOR_H_
#define DLR_ALLOCATOR_H_

#include <memory>

#ifndef DLR_ALLOC_TYPEDEF
#define DLR_ALLOC_TYPEDEF
/*! \brief A pointer to a malloc-like function. */
typedef void* (*DLRMallocFunctionPtr)(size_t);
/*! \brief A pointer to a free-like function. */
typedef void (*DLRFreeFunctionPtr)(void*);
/*! \brief A pointer to a memalign-like function. */
typedef void* (*DLRMemalignFunctionPtr)(size_t, size_t);
#endif

namespace dlr {

/*! \brief Stores custom allocation functions. */
class DLRAllocatorFunctions {
 private:
  /*! \brief Custom malloc-like function, nullptr if not set. */
  static DLRMallocFunctionPtr malloc_fn_;

  /*! \brief Custom free-like function, nullptr if not set. */
  static DLRFreeFunctionPtr free_fn_;

  /*! \brief Custom memalign-like function, nullptr if not set. */
  static DLRMemalignFunctionPtr memalign_fn_;

 public:
  /*! \brief Set global allocator malloc function. */
  static void SetMallocFunction(DLRMallocFunctionPtr malloc_fn);

  /*! \brief Set global allocator free function. */
  static void SetFreeFunction(DLRFreeFunctionPtr free_fn);

  /*! \brief Set global allocator memalign function. */
  static void SetMemalignFunction(DLRMemalignFunctionPtr memalign_fn);

  /*! \brief Get current malloc function pointer, returns nullptr if not set. */
  static DLRMallocFunctionPtr GetMallocFunction();

  /*! \brief Get current free function pointer, returns nullptr if not set. */
  static DLRFreeFunctionPtr GetFreeFunction();

  /*! \brief Get current memalign function pointer, returns nullptr if not set. */
  static DLRMemalignFunctionPtr GetMemalignFunction();

  /*! \brief Clear global allocator functions. */
  static void Clear();

  /*! \brief Check if all global allocator functions are set. */
  static bool AllSet();

  /*! \brief Check if any global allocator functions are set. */
  static bool AnySet();

  /*! \brief Allocate data, using custom allocator if set, otherwise use malloc. */
  static void* Malloc(size_t size);

  /*! \brief Free data, using custom free if set, otherwise use free. */
  static void Free(void* ptr);
};

/*! \brief STL-compatible allocator using allocator functions from DLRAllocatorFunctions. */
template <typename T>
class DLRAllocator : public std::allocator<T> {
 private:
  using Base = std::allocator<T>;
  using Pointer = typename std::allocator_traits<Base>::pointer;
  using SizeType = typename std::allocator_traits<Base>::size_type;

 public:
  DLRAllocator() = default;

  template <typename U>
  DLRAllocator(const DLRAllocator<U>& a) : Base(a) {}

  template <typename U>
  struct rebind {
    using other = DLRAllocator<U>;
  };

  Pointer allocate(SizeType n) {
    if (DLRAllocatorFunctions::GetMallocFunction()) {
      return static_cast<T*>(DLRAllocatorFunctions::Malloc(n * sizeof(T)));
    }
    return Base::allocate(n);
  }

  void deallocate(Pointer p, SizeType n) {
    if (DLRAllocatorFunctions::GetFreeFunction()) {
      DLRAllocatorFunctions::Free(p);
      return;
    }
    Base::deallocate(p, n);
  }
};

/*! \brief ostringstream which uses the custom allocators. */
typedef std::basic_ostringstream<char, std::char_traits<char>, DLRAllocator<char>> DLRStringStream;

}  // namespace dlr

#endif  // DLR_ALLOCATOR_H_
