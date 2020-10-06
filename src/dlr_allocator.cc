#include "dlr_allocator.h"

namespace dlr {

DLRMallocFunctionPtr DLRAllocatorFunctions::malloc_fn_ = nullptr;
DLRReallocFunctionPtr DLRAllocatorFunctions::realloc_fn_ = nullptr;
DLRFreeFunctionPtr DLRAllocatorFunctions::free_fn_ = nullptr;
DLRMemalignFunctionPtr DLRAllocatorFunctions::memalign_fn_ = nullptr;

void DLRAllocatorFunctions::Set(DLRMallocFunctionPtr malloc_fn, DLRReallocFunctionPtr realloc_fn,
                                DLRFreeFunctionPtr free_fn, DLRMemalignFunctionPtr memalign_fn) {
  // realloc is not used yet, so it is not checked.
  if (!malloc_fn) throw dmlc::Error("Custom malloc was not set.");
  if (!free_fn) throw dmlc::Error("Custom free was not set.");
  if (!memalign_fn) throw dmlc::Error("Custom memalign was not set.");
  malloc_fn_ = malloc_fn;
  realloc_fn_ = realloc_fn;
  free_fn_ = free_fn;
  memalign_fn_ = memalign_fn;
}

DLRMallocFunctionPtr DLRAllocatorFunctions::GetMallocFunction() { return malloc_fn_; }

DLRReallocFunctionPtr DLRAllocatorFunctions::GetReallocFunction() { return realloc_fn_; }

DLRFreeFunctionPtr DLRAllocatorFunctions::GetFreeFunction() { return free_fn_; }

DLRMemalignFunctionPtr DLRAllocatorFunctions::GetMemalignFunction() { return memalign_fn_; }

void DLRAllocatorFunctions::Clear() {
  malloc_fn_ = nullptr;
  realloc_fn_ = nullptr;
  free_fn_ = nullptr;
  memalign_fn_ = nullptr;
}

bool DLRAllocatorFunctions::IsSet() {
  return malloc_fn_ != nullptr && free_fn_ != nullptr && memalign_fn_ != nullptr;
}

void* DLRAllocatorFunctions::Malloc(size_t size) {
  return IsSet() ? (*malloc_fn_)(size) : malloc(size);
}

void DLRAllocatorFunctions::Free(void* ptr) {
  if (IsSet()) {
    (*free_fn_)(ptr);
  } else {
    free(ptr);
  }
}

}  // namespace dlr
