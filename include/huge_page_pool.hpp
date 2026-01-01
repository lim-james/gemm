#pragma once

#include <cstdint>
#include <sys/mman.h>
#include <print>

template<std::size_t PAGE_COUNT> 
class huge_page_pool {
private:

    static constexpr std::size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
    static constexpr std::size_t TOTAL_SIZE = HUGE_PAGE_SIZE * PAGE_COUNT;

    void* base_ptr_;
    std::size_t offset_;

public:
    
    huge_page_pool() {
        constexpr int fd = -1;
        constexpr __off_t offset = 0;
        base_ptr_ = mmap(
            nullptr, TOTAL_SIZE, 
            PROT_READ | PROT_WRITE, 
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            fd, offset
        );
        
        if (base_ptr_ == MAP_FAILED) 
            return;

        for (std::size_t i{}; i < TOTAL_SIZE; i += 4096)
            ((char*)base_ptr_)[i] = 0;
    }

    template<typename T>
    T* get(std::size_t n) {
        const std::size_t size = (n * sizeof(T) + 63) & ~63;
        
        if (offset_ + size > TOTAL_SIZE)  {
            std::print("SIGMA");
            return nullptr;
        }

        void* ptr = (char*)base_ptr_ + offset_;
        offset_ += size;
        return static_cast<T*>(ptr);
    }

    template<typename T>
    void release(std::size_t n) {
        const std::size_t size = (n * sizeof(T) + 63) & ~63;
        offset_ -= size;
    }

    ~huge_page_pool() {
        munmap(base_ptr_, TOTAL_SIZE);
    }
};
