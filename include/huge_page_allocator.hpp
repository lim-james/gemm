#pragma once 

#include "huge_page_pool.hpp"
#include <cstdint>
#include <type_traits>
#include <print>

template<typename T>
struct huge_page_allocator {
private:

    static huge_page_pool<512> pool_;

public:

    using value_type = T;

    huge_page_allocator() noexcept = default;
    
    template<typename U>
    huge_page_allocator(const huge_page_allocator<U>&) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return pool_.get<T>(n);
    }
    
    void deallocate(T*, std::size_t n) noexcept {
        return pool_.release<T>(n);
    }

    template<typename U>
    struct rebind { using other = huge_page_allocator<U>; };

    using is_always_equal = std::true_type;
};

template <typename T>
huge_page_pool<512> huge_page_allocator<T>::pool_{};
