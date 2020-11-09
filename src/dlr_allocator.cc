#include "dlr_allocator.h"

namespace dlr {

DLRMallocFunctionPtr DLRAllocatorFunctions::malloc_fn_ = nullptr;
DLRFreeFunctionPtr DLRAllocatorFunctions::free_fn_ = nullptr;
DLRMemalignFunctionPtr DLRAllocatorFunctions::memalign_fn_ = nullptr;

void DLRAllocatorFunctions::SetMallocFunction(DLRMallocFunctionPtr malloc_fn) {
  malloc_fn_ = malloc_fn;
}

void DLRAllocatorFunctions::SetFreeFunction(DLRFreeFunctionPtr free_fn) { free_fn_ = free_fn; }

void DLRAllocatorFunctions::SetMemalignFunction(DLRMemalignFunctionPtr memalign_fn) {
  memalign_fn_ = memalign_fn;
}

DLRMallocFunctionPtr DLRAllocatorFunctions::GetMallocFunction() { return malloc_fn_; }

DLRFreeFunctionPtr DLRAllocatorFunctions::GetFreeFunction() { return free_fn_; }

DLRMemalignFunctionPtr DLRAllocatorFunctions::GetMemalignFunction() { return memalign_fn_; }

void DLRAllocatorFunctions::Clear() {
  malloc_fn_ = nullptr;
  free_fn_ = nullptr;
  memalign_fn_ = nullptr;
}

bool DLRAllocatorFunctions::AllSet() {
  return malloc_fn_ != nullptr && free_fn_ != nullptr && memalign_fn_ != nullptr;
}

bool DLRAllocatorFunctions::AnySet() {
  return malloc_fn_ != nullptr || free_fn_ != nullptr || memalign_fn_ != nullptr;
}

void* DLRAllocatorFunctions::Malloc(size_t size) {
  return malloc_fn_ ? (*malloc_fn_)(size) : malloc(size);
}

void DLRAllocatorFunctions::Free(void* ptr) {
  if (free_fn_) {
    (*free_fn_)(ptr);
  } else {
    free(ptr);
  }
}

}  // namespace dlr
