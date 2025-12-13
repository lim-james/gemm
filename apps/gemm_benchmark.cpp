#include "mat.hpp"
#include <benchmark/benchmark.h>
#include <cmath> 

template <std::size_t N>
void RunBenchmark(benchmark::State& state, Impl implementation) {
    static auto a = SquareMatrix<std::int32_t, N>::make_random(1, 10);
    static auto b = SquareMatrix<std::int32_t, N>::make_random(1, 10);

    for (auto _ : state) {
        auto result = a.multiply(b, implementation);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    double ops = 2.0 * std::pow(N, 3);
    
    state.counters["GOps"] = benchmark::Counter(
        ops, 
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000
    );
    
    double bytes = 3.0 * std::pow(N, 2) * sizeof(std::int32_t);
    state.counters["Bandwidth"] = benchmark::Counter(
        bytes, 
        benchmark::Counter::kIsRate | benchmark::Counter::kAvgThreads,
        benchmark::Counter::kIs1000
    );
}

template <std::size_t N>
static void BM_Naive(benchmark::State& state) {
    RunBenchmark<N>(state, Impl::NAIVE);
}

template <std::size_t N>
static void BM_Transposed(benchmark::State& state) {
    RunBenchmark<N>(state, Impl::TRANSPOSED);
}

template <std::size_t N>
static void BM_Tiling(benchmark::State& state) {
    RunBenchmark<N>(state, Impl::TILING);
}

#define REGISTER_SIZE(N) \
    BENCHMARK(BM_Naive<N>)->Name("Naive/" #N); \
    BENCHMARK(BM_Transposed<N>)->Name("Tranposed/" #N); \
    BENCHMARK(BM_Tiling<N>)->Name("Tiling/" #N);

REGISTER_SIZE(4);
REGISTER_SIZE(8);
REGISTER_SIZE(16);
REGISTER_SIZE(32);
REGISTER_SIZE(64);
REGISTER_SIZE(128);

BENCHMARK_MAIN();
