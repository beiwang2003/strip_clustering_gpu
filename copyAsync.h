#ifndef HeterogeneousCore_CUDAUtilities_copyAsync_h
#define HeterogeneousCore_CUDAUtilities_copyAsync_h

#include "device_unique_ptr.h"
#include "host_unique_ptr.h"

#include <cuda_runtime.h>

#include <type_traits>

namespace cudautils {
  // Single element
  template <typename T>
  inline
  void copyAsync(cudautils::device::unique_ptr<T>& dst, const cudautils::host::unique_ptr<T>& src, cudaStream_t stream) {
    // Shouldn't compile for array types because of sizeof(T), but
    // let's add an assert with a more helpful message
    static_assert(std::is_array<T>::value == false, "For array types, use the other overload with the size parameter");
    cudaMemcpyAsync(dst.get(), src.get(), sizeof(T), cudaMemcpyDefault, stream);
  }

  template <typename T>
  inline
  void copyAsync(cudautils::host::unique_ptr<T>& dst, const cudautils::device::unique_ptr<T>& src, cudaStream_t stream) {
    static_assert(std::is_array<T>::value == false, "For array types, use the other overload with the size parameter");
    cudaMemcpyAsync(dst.get(), src.get(), sizeof(T), cudaMemcpyDefault, stream);
  }

  // Multiple elements
  template <typename T>
  inline
  void copyAsync(cudautils::device::unique_ptr<T[]>& dst, const cudautils::host::unique_ptr<T[]>& src, size_t nelements, cudaStream_t stream) {
    cudaMemcpyAsync(dst.get(), src.get(), nelements*sizeof(T), cudaMemcpyDefault, stream);
  }

  template <typename T>
  inline
  void copyAsync(cudautils::host::unique_ptr<T[]>& dst, const cudautils::device::unique_ptr<T[]>& src, size_t nelements, cudaStream_t stream) {
    cudaMemcpyAsync(dst.get(), src.get(), nelements*sizeof(T), cudaMemcpyDefault, stream);
  }
}

#endif
