#pragma once 

#include <cstddef>
#include <new>
#include <type_traits>

template<typename T, std::size_t Alignment>
struct aligned_allocator {
    static_assert(Alignment >= sizeof(T));
    static_assert((Alignment & (Alignment - 1)) == 0);

    using value_type = T;

    aligned_allocator() noexcept = default;
    
    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        void* p = ::operator new(bytes, std::align_val_t(Alignment));
        return static_cast<T*>(p);
    }
    
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Alignment));
    }

    template<typename U>
    struct rebind { using other = aligned_allocator<U, Alignment>; };

    using is_always_equal = std::true_type;
};
