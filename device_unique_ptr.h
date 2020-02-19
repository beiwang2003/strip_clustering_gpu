#ifndef HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h

#include <memory>
#include <functional>
#include <cassert>

#include "allocate_device.h"

namespace cudautils {
  namespace device {
    namespace impl {
      // Additional layer of types to distinguish from host::unique_ptr
      class DeviceDeleter {
      public:
        DeviceDeleter() = default; // for edm::Wrapper
        DeviceDeleter(int device): device_{device} {}

        void operator()(void *ptr) {
          assert(device_ >= 0 || ptr == nullptr);
          if(__builtin_expect(device_ >= 0, true)) {
            free_device(device_, ptr);
          }
        }
      private:
        int device_ = -1;
      };
    }

    template <typename T>
    using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

    namespace impl {
      template <typename T>
      struct make_device_unique_selector { using non_array = cudautils::device::unique_ptr<T>; };
      template <typename T>
      struct make_device_unique_selector<T[]> { using unbounded_array = cudautils::device::unique_ptr<T[]>; };
      template <typename T, size_t N>
      struct make_device_unique_selector<T[N]> { struct bounded_array {}; };
    }
  }

  template <typename T>
  typename device::impl::make_device_unique_selector<T>::non_array
  make_device_unique(cudaStream_t stream) {
    static_assert(std::is_trivially_constructible<T>::value, "Allocating with non-trivial constructor on the device memory is not supported");
    int dev;
    cudaGetDevice(&dev);
    void *mem = cudautils::allocate_device(dev, sizeof(T), stream);
    return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                            device::impl::DeviceDeleter{dev}};
  }

  template <typename T>
  typename device::impl::make_device_unique_selector<T>::unbounded_array
  make_device_unique(size_t n, cudaStream_t stream) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value, "Allocating with non-trivial constructor on the device memory is not supported");
    int dev;
    cudaGetDevice(&dev);
    void *mem = cudautils::allocate_device(dev, n*sizeof(element_type), stream);
    return typename device::impl::make_device_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                  device::impl::DeviceDeleter{dev}};
  }

  template <typename T, typename ...Args>
  typename device::impl::make_device_unique_selector<T>::bounded_array
  make_device_unique(Args&&...) = delete;


  // No check for the trivial constructor, make it clear in the interface
  template <typename T>
  typename device::impl::make_device_unique_selector<T>::non_array
  make_device_unique_uninitialized(cudaStream_t stream) {
    int dev;
    cudaGetDevice(&dev);
    void *mem = cudautils::allocate_device(dev, sizeof(T), stream);
    return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                            device::impl::DeviceDeleter{dev}};
  }

  template <typename T>
  typename device::impl::make_device_unique_selector<T>::unbounded_array
  make_device_unique_uninitialized(size_t n, cudaStream_t stream) {
    using element_type = typename std::remove_extent<T>::type;
    int dev;
    cudaGetDevice(&dev);
    void *mem = cudautils::allocate_device(dev, n*sizeof(element_type), stream);
    return typename device::impl::make_device_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                  device::impl::DeviceDeleter{dev}};
  }

  template <typename T, typename ...Args>
  typename device::impl::make_device_unique_selector<T>::bounded_array
  make_device_unique_uninitialized(Args&&...) = delete;
}

#endif
