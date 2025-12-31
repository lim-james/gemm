mkdir -p results

echo "Starting benchmarks on independent cores..."

sudo cpupower frequency-set -g performance
echo "[Locked CPU Frequency Scaling]"

# Run SSE2 Benchmark on Core 0
taskset -c 0 ./build/gemm_benchmark_no_avx --benchmark_format=csv > results/benchmark_sse2.csv &
PID_SSE_BENCH=$!
echo "  [Core 0] Dispatched SSE2 (PID: $PID_SSE_BENCH)"

# Run AVX2 Benchmark on Core 2 
taskset -c 2 ./build/gemm_benchmark_avx2 --benchmark_format=csv > results/benchmark_avx2.csv &
PID_AVX_BENCH=$!
echo "  [Core 2] Dispatched AVX2 (PID: $PID_AVX_BENCH)"

# Run SSE2 Perf on Core 4
taskset -c 4 ./build/perf_driver_no_avx > results/perf_sse2.txt &
PID_SSE_PERF=$!

echo "  [Core 4] Dispatched SSE2 (PID: $PID_SSE_PERF)"

# Run AVX2 Perf on Core 6
taskset -c 6 ./build/perf_driver_avx2 > results/perf_avx2.txt &
PID_AVX_PERF=$!

echo "  [Core 6] Dispatched AVX (2PID: $PID_AVX_PERF)"

echo "Waiting for benchmarks to complete..."

wait $PID_SSE_BENCH
wait $PID_AVX_BENCH
wait $PID_SSE_PERF
wait $PID_AVX_PERF


echo "benchmarks and perf finished. Results saved in ./results/"

sudo cpupower frequency-set -g powersave
echo "[Relaxed CPU Frequency Scaling]"
