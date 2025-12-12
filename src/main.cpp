#include "mat.hpp"

#include <algorithm>
#include <print>
#include <chrono>
#include <format>
#include <cassert>
#include <vector>
#include <fstream>
#include <filesystem>

template<typename ...Args>
inline void log_row(Args... args) {
    std::println("{:5} | {:4} | {:10} | {:10} | {:5}", args...);
}

class [[nodiscard]] ScopeTimer {
private:
    double* out_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;

public:
    ScopeTimer(double* out)
        : out_(out)
        , start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopeTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        *out_ = std::chrono::duration<double, std::milli>(end - start_).count();
    }
};


constexpr double calculate_throughput_per_s(double tottime_ms, std::size_t ncalls) {
    constexpr double ms_to_s = 1'000.0;
    return static_cast<double>(ncalls) / tottime_ms * ms_to_s; 
}

constexpr double calculate_throughput_per_ms(double tottime_ms, std::size_t ncalls) {
    return static_cast<double>(ncalls) / tottime_ms; 

}

template<typename Fn>
double run_batch(const Fn& fn, std::size_t batch_size) {
    double tottime_ms;
    
    {
        auto _ = ScopeTimer(&tottime_ms);
        for (std::size_t i = 0; i < batch_size; ++i) 
            fn();
    }

    return tottime_ms;
}

template<typename Fn>
std::vector<double> run_trail(
    const Fn& fn, 
    std::size_t batch_size, 
    std::size_t num_trail
) {
    std::vector<double> sample_timings;
    sample_timings.reserve(num_trail);

    std::println("----- RUNNING -----");
    std::println("Num trail:  {}", num_trail);
    std::println("Batch size: {}", batch_size);
    std::println("-------------------");

    for (std::size_t i = 0; i < num_trail; ++i) 
        sample_timings.push_back(run_batch(fn, batch_size));

    return sample_timings;
}

template<std::size_t N>
void run_matrix_multiplication_simd() {
    static auto a = SquareMatrix<std::int32_t, N>::make_random(1, 10);
    static auto b = SquareMatrix<std::int32_t, N>::make_random(1, 10);
    volatile auto sink = a.mul_simd(b);
}

template<std::size_t N>
void run_matrix_multiplication_naive() {
    static auto a = SquareMatrix<std::int32_t, N>::make_random(1, 10);
    static auto b = SquareMatrix<std::int32_t, N>::make_random(1, 10);
    volatile auto sink = a.mul_naive(b);
}

void run_simd_experiment(int matrix_width) {
    switch (matrix_width) {
    case 4:   run_matrix_multiplication_simd<4>();   break;
    case 8:   run_matrix_multiplication_simd<8>();   break;
    case 16:  run_matrix_multiplication_simd<16>();  break;
    case 32:  run_matrix_multiplication_simd<32>();  break;
    case 64:  run_matrix_multiplication_simd<64>();  break;
    case 128: run_matrix_multiplication_simd<128>(); break;
    }
}

void run_naive_experiment(int matrix_width) {
    switch (matrix_width) {
    case 4:   run_matrix_multiplication_naive<4>();   break;
    case 8:   run_matrix_multiplication_naive<8>();   break;
    case 16:  run_matrix_multiplication_naive<16>();  break;
    case 32:  run_matrix_multiplication_naive<32>();  break;
    case 64:  run_matrix_multiplication_naive<64>();  break;
    case 128: run_matrix_multiplication_naive<128>(); break;
    }
}

void save_runtimes(std::filesystem::path path, const std::vector<double>& times) {
    std::ofstream file(path);
    for (double t: times)
        file << t << '\n';
}

int main(int argsc, char** argsv) {
    if (argsc <= 1) {
        return 0;
    }

    const int matrix_width = std::stoi(argsv[1]);
    const int batch_size   = argsc > 2 ? std::stoi(argsv[2]) : 1'000;
    const int num_trail    = argsc > 3 ? std::stoi(argsv[3]) : 1'000;

    run_naive_experiment(matrix_width);
    auto naive_timings = run_trail(
        [matrix_width]() { run_naive_experiment(matrix_width); }, 
        batch_size, 
        num_trail
    );

    save_runtimes(
        std::format("mat{}_naive_{}x{}.txt", matrix_width, batch_size, num_trail),
        naive_timings
    );

    run_simd_experiment(matrix_width);
    auto simd_timings = run_trail(
        [matrix_width]() { run_simd_experiment(matrix_width); }, 
        batch_size, 
        num_trail
    );

    save_runtimes(
        std::format("mat{}_simd_{}x{}.txt", matrix_width, batch_size, num_trail),
        simd_timings
    );

    
    return 0;
}

