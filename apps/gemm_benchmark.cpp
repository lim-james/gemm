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
    BENCHMARK(RunBenchmark<N, Impl::NAIVE>)           ->Name("Naive/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TRANSPOSED>)      ->Name("Tranposed/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TRANSPOSED_SIMD>) ->Name("Tranposed Simd/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED>)           ->Name("Tiled/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED_SIMD>)      ->Name("Tiled SIMD/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED_PREFETCH>)  ->Name("Tiled PREFETCH/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED_REGISTERS>)  ->Name("Tiled REGISTERS/" #N); 

#define REGISTER_LARGE_SIZE(N) \
    BENCHMARK(RunBenchmark<N, Impl::TRANSPOSED>)      ->Name("Tranposed/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TRANSPOSED_SIMD>) ->Name("Tranposed Simd/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED>)           ->Name("Tiled/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED_SIMD>)      ->Name("Tiled SIMD/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED_PREFETCH>)  ->Name("Tiled PREFETCH/" #N); \
    BENCHMARK(RunBenchmark<N, Impl::TILED_REGISTERS>)  ->Name("Tiled REGISTERS/" #N); 


// REGISTER_SIZE(4);
REGISTER_SIZE(8);
REGISTER_SIZE(16);
REGISTER_SIZE(32);
REGISTER_SIZE(64);
REGISTER_SIZE(128);
REGISTER_SIZE(256);
REGISTER_SIZE(512);
REGISTER_LARGE_SIZE(1024);
REGISTER_LARGE_SIZE(2048);
REGISTER_LARGE_SIZE(4096);
REGISTER_LARGE_SIZE(8192);

BENCHMARK_MAIN();


/*
Tiling/984   186284486 ns    186237021 ns            4 Bandwidth=15.5972M/s GOps=2.55793G/s
Tiling/992   165485639 ns    165455378 ns            4 Bandwidth=17.8428M/s GOps=2.95001G/s
Tiling/1000  216073456 ns    216009561 ns            3 Bandwidth=18.5177M/s GOps=3.08628G/s
Tiling/1008  167179437 ns    167151276 ns            4 Bandwidth=18.2361M/s GOps=3.06367G/s
Tiling/1016  144411755 ns    144386218 ns            5 Bandwidth=17.1582M/s GOps=2.90546G/s

Tiling/1024  112496823 ns    112444421 ns            6 Bandwidth=18.6506M/s GOps=3.18303G/s

Tiling/1032  155241225 ns    155216042 ns            4 Bandwidth=20.5847M/s GOps=3.54056G/s
Tiling/1040  157427774 ns    157394337 ns            4 Bandwidth=20.6157M/s GOps=3.57339G/s
Tiling/1048  167278248 ns    167257346 ns            4 Bandwidth=19.6997M/s GOps=3.44087G/s
Tiling/1056  120408128 ns    120388858 ns            6 Bandwidth=18.5256M/s GOps=3.2605G/s
Tiling/1064  177535558 ns    177505632 ns            4 Bandwidth=19.1334M/s GOps=3.39299G/s
*/
