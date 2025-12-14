#include "mat.hpp"
#include <benchmark/benchmark.h>
#include <cmath> 

template <std::size_t N, Impl IMPLEMENTATION>
void RunBenchmark(benchmark::State& state) {
    static auto a = SquareMatrix<std::int32_t, N>::make_random(1, 10);
    static auto b = SquareMatrix<std::int32_t, N>::make_random(1, 10);

    SquareMatrix<std::int32_t, N> result{};
    for (auto _ : state) {
        a.multiply(b, result, IMPLEMENTATION);
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

#define REGISTER_SIZE(N) \
    BENCHMARK(RunBenchmark<N, Impl::NAIVE>)     ->Name("Naive/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TRANSPOSED>)->Name("Tranposed/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::SIMD>)      ->Name("Simd/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILING>)    ->Name("Tiling/" #N);


REGISTER_SIZE(4);
REGISTER_SIZE(8);
REGISTER_SIZE(16);
REGISTER_SIZE(32);
REGISTER_SIZE(64);
REGISTER_SIZE(128);
REGISTER_SIZE(256);
REGISTER_SIZE(512);
REGISTER_SIZE(1024);
REGISTER_SIZE(2048);
REGISTER_SIZE(4096);

BENCHMARK_MAIN();
