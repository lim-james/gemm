#include "mat.hpp"

#include <memory>
#include <print>
#include <vector>
#include <utility>
#include <cstring>
#include <cstdint>
#include <optional>

#include <emmintrin.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <asm/unistd.h>

class [[nodiscard]] PerfEvent {
private:
    static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                                int cpu, int group_fd, unsigned long flags) {
        return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    }

    int fd_ = -1;
    long long* result_;
    PerfEvent(int fd, long long* result): fd_(fd), result_(result) {
        ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
    }

public:

    PerfEvent(const PerfEvent&) = delete;
    PerfEvent& operator=(const PerfEvent&) = delete;

    PerfEvent(PerfEvent&& other): fd_(other.fd_), result_(other.result_) {
        other.fd_ = -1;
        other.result_ = nullptr;
    }

    PerfEvent& operator=(PerfEvent&& other) {
        if (this == &other)
            return *this;

        fd_ = std::exchange(other.fd_, -1);
        result_ = std::exchange(other.result_, nullptr);

        return *this;
    }

    ~PerfEvent() noexcept {
        if (fd_ != -1) {
            ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
            read(fd_, result_, sizeof(long long));
            close(fd_);
        }
    }

    static std::optional<PerfEvent> make_event(
        std::uint32_t type, 
        std::uint64_t config,
        long long* result
    ) {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(pe));
        pe.type = type;
        pe.size = sizeof(pe);
        pe.config = config; 
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        int fd = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd == -1) {
            std::println("Error opening perf event");
            return std::nullopt;
        }

        return PerfEvent(fd, result);
    }
};


struct PerfResults {
    long long l1d_misses;
    long long llc_misses;
    long long instructions;
    long long cycles;
};

template<typename T, std::size_t N>
PerfResults get_perf_results(Impl implementation) {
    constexpr std::size_t CACHELINE = 64;
    auto a = SquareMatrix<T, N>::make_random(1, 10);
    auto b = SquareMatrix<T, N>::make_random(1, 10);
    SquareMatrix<T, N> result{};

    constexpr std::size_t FLUSH_SIZE = N * N * sizeof(T);

    auto* a_ptr  = a.data();
    auto* at_ptr = a.data_transposed();
    auto* b_ptr  = b.data();
    auto* bt_ptr = b.data_transposed();
    auto* r_ptr  = result.data();
    auto* rt_ptr = result.data_transposed();

    for (std::size_t i = 0; i < FLUSH_SIZE; i += CACHELINE) {
        _mm_clflush(reinterpret_cast<const void*>(reinterpret_cast<const char*>(a_ptr)+i)); 
        _mm_clflush(reinterpret_cast<const void*>(reinterpret_cast<const char*>(at_ptr)+i)); 
        _mm_clflush(reinterpret_cast<const void*>(reinterpret_cast<const char*>(b_ptr)+i)); 
        _mm_clflush(reinterpret_cast<const void*>(reinterpret_cast<const char*>(bt_ptr)+i)); 
        _mm_clflush(reinterpret_cast<const void*>(reinterpret_cast<const char*>(r_ptr)+i)); 
        _mm_clflush(reinterpret_cast<const void*>(reinterpret_cast<const char*>(rt_ptr)+i)); 
    }
    _mm_mfence();

    PerfResults perf_results{};

    {
        auto perf_l1d = PerfEvent::make_event(
            PERF_TYPE_HW_CACHE, 
            (PERF_COUNT_HW_CACHE_L1D) |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
            &perf_results.l1d_misses
        );

        auto perf_llc = PerfEvent::make_event(
            PERF_TYPE_HARDWARE,
            PERF_COUNT_HW_CACHE_MISSES,
            &perf_results.llc_misses
        );

        auto perf_instr = PerfEvent::make_event(
            PERF_TYPE_HARDWARE,
            PERF_COUNT_HW_INSTRUCTIONS,
            &perf_results.instructions
        );

        auto perf_cycles = PerfEvent::make_event(
            PERF_TYPE_HARDWARE,
            PERF_COUNT_HW_CPU_CYCLES,
            &perf_results.cycles
        );

        if (perf_l1d && perf_llc && perf_instr && perf_cycles) {
            a.multiply(b, result, implementation);
        } 
    }

    return perf_results;
}

template<std::size_t N>
void perf_size() {
    using T = std::int32_t;
    auto transposed = get_perf_results<T, N>(Impl::TRANSPOSED);
    auto simd       = get_perf_results<T, N>(Impl::SIMD);
    auto tiling     = get_perf_results<T, N>(Impl::TILING);

    if constexpr (N < 1024) {
        auto naive      = get_perf_results<T, N>(Impl::NAIVE);
        std::println("{:4} | NAIVE      | {:11} | {:11} | {:11} | {:11}", N, naive.l1d_misses, naive.llc_misses, naive.instructions, naive.cycles);
    }
    std::println("{:4} | TRANSPOSED | {:11} | {:11} | {:11} | {:11}", N, transposed.l1d_misses, transposed.llc_misses, transposed.instructions, transposed.cycles);
    std::println("{:4} | SIMD       | {:11} | {:11} | {:11} | {:11}", N, simd.l1d_misses, simd.llc_misses, simd.instructions, simd.cycles);
    std::println("{:4} | TILING     | {:11} | {:11} | {:11} | {:11}", N, tiling.l1d_misses, tiling.llc_misses, tiling.instructions, tiling.cycles);
}

int main() {
    std::println("SIZE | METHOD     | {:11} | {:11} | {:11} | {:11}", "L1D MISSES", "LLC MISSES", "INSTR", "CPU CYCLES");
    for (int i{}; i < 100; ++i) {
        perf_size<4>(); 
        perf_size<8>(); 
        perf_size<16>(); 
        perf_size<32>(); 
        perf_size<64>(); 
        perf_size<128>(); 
        perf_size<256>(); 
        perf_size<512>(); 
        perf_size<1024>(); 
        perf_size<2048>(); 
    }

    return 0;
}
