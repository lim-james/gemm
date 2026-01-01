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
    long long tlb_misses;
    long long page_faults;
    long long instructions;
    long long cycles;
    long long stalls;
    long long clock;
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

        auto perf_tlb = PerfEvent::make_event(
            PERF_TYPE_HW_CACHE, 
            (PERF_COUNT_HW_CACHE_DTLB) |
            (PERF_COUNT_HW_CACHE_OP_READ << 8) |
            (PERF_COUNT_HW_CACHE_RESULT_MISS << 16),
            &perf_results.tlb_misses
        );

        auto perf_page_faults = PerfEvent::make_event(
            PERF_TYPE_SOFTWARE,
            PERF_COUNT_SW_PAGE_FAULTS,
            &perf_results.page_faults
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

        auto perf_clock = PerfEvent::make_event(
            PERF_TYPE_SOFTWARE,
            PERF_COUNT_SW_CPU_CLOCK,
            &perf_results.clock
        );

        constexpr std::uint64_t AMD_BACKEND_STALL_CYCLES = 0x05f;
        auto perf_stalls = PerfEvent::make_event(
            PERF_TYPE_RAW,
            AMD_BACKEND_STALL_CYCLES,
            &perf_results.stalls
        );

        if (perf_l1d   && perf_llc    && perf_tlb    && perf_page_faults && 
            perf_instr && perf_cycles && perf_stalls && perf_clock) {
            a.multiply(b, result, implementation);
        } 
    }

    return perf_results;
}

void print_row(const std::string& method, std::size_t N, const PerfResults& results) {
    std::println("| {:4} | {:13} | {:11} | {:11} | {:11} | {:11} | {:11} | {:11} | {:11} | {:11} |", 
        N, 
        method,
        results.l1d_misses,   results.llc_misses, results.tlb_misses, results.page_faults,
        results.instructions, results.cycles,     results.stalls,     results.clock
    );
}

template<std::size_t N>
void perf_size() {
    using T = std::int32_t;
    auto transposed     = get_perf_results<T, N>(Impl::TRANSPOSED);
    auto simd           = get_perf_results<T, N>(Impl::TRANSPOSED_SIMD);
    auto tiled          = get_perf_results<T, N>(Impl::TILED);
    auto tiled_simd     = get_perf_results<T, N>(Impl::TILED_SIMD);
    auto tiled_prefetch = get_perf_results<T, N>(Impl::TILED_PREFETCH);
    auto tiled_reg      = get_perf_results<T, N>(Impl::TILED_REGISTERS);

    if constexpr (N < 1024) {
        auto naive      = get_perf_results<T, N>(Impl::NAIVE);
        print_row("NAIVE", N, naive);
    }

    if constexpr (N < 8192) {
        print_row("TRANSPOSED",    N, transposed);
        print_row("SIMD",          N, simd);
    }

    print_row("TILED",         N, tiled);
    print_row("TILED_SIMD",    N, tiled_simd);
    print_row("TILED_FETCHED", N, tiled_prefetch);
    print_row("TILED_REG",     N, tiled_reg);
}

int main() {
    std::println("| {:4} | {:13} | {:11} | {:11} | {:11} | {:11} | {:11} | {:11} | {:11} | {:11} |", 
        "SIZE", "METOHD",
        "L1D MISSES", "LLC MISSES", "TLB MISSES", "PAGE FAULTS", 
        "INSTR",     "CPU CYCLES", "STALLS",     "CLOCK"
    );
    for (int i{}; i < 1; ++i) {
        // perf_size<4>(); 
        perf_size<2 << 3>(); 
        perf_size<2 << 4>(); 
        perf_size<2 << 5>(); 
        perf_size<2 << 6>(); 
        perf_size<2 << 7>(); 
        perf_size<2 << 8>(); 
        perf_size<2 << 9>(); 
        perf_size<2 << 10>(); 
        perf_size<2 << 11>(); 
        perf_size<2 << 12>(); 
        perf_size<2 << 13>(); 
        perf_size<2 << 14>(); 
        perf_size<2 << 15>(); 
    }

    return 0;
}
