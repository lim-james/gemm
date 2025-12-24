#pragma once 

#include <cstddef>
#include <new>

template <class T, std::size_t Alignment>
struct aligned_allocator {
  using value_type = T;

  aligned_allocator() noexcept = default;
  template <class U>
  aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    std::size_t bytes = n * sizeof(T);
    void* p = ::operator new(bytes, std::align_val_t(Alignment));
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t(Alignment));
  }

  template <class U>
  struct rebind { using other = aligned_allocator<U, Alignment>; };
};

template <class T1, std::size_t A1, class T2, std::size_t A2>
bool operator==(const aligned_allocator<T1, A1>&, const aligned_allocator<T2, A2>&) { return A1 == A2; }

template <class T1, std::size_t A1, class T2, std::size_t A2>
bool operator!=(const aligned_allocator<T1, A1>& a, const aligned_allocator<T2, A2>& b) { return !(a == b); }
