#include "mat.hpp"

#include <algorithm>
#include <print>
#include <array>
#include <string_view>
#include <cassert>
#include <cstdint>

template<std::size_t N>
bool validate_matrix_multiplication(
    std::int32_t lower_bound, 
    std::int32_t upper_bound,
    Impl implementation
) {
    auto a = SquareMatrix<std::int32_t, N>::make_random(lower_bound, upper_bound);
    auto b = SquareMatrix<std::int32_t, N>::make_random(lower_bound, upper_bound);
    
    SquareMatrix<std::int32_t, N> out1{}; a.multiply(b, out1, Impl::NAIVE);
    SquareMatrix<std::int32_t, N> out2{}; a.multiply(b, out2, implementation);

    return out1 == out2;
}

template<std::size_t START_SIZE, std::size_t END_SIZE>
std::size_t validate_range_of_matrices(
    std::int32_t lower_bound, 
    std::int32_t upper_bound,
    Impl implementation
) {
    std::size_t correct_count = static_cast<std::size_t>(
        validate_matrix_multiplication<START_SIZE>(lower_bound, upper_bound, implementation)
    );

    if constexpr (START_SIZE < END_SIZE)
        correct_count += validate_range_of_matrices<START_SIZE + 4, END_SIZE>(
            lower_bound, upper_bound, implementation
        );

    return correct_count;
}

std::size_t validate_implementation(
    std::size_t  num_runs,
    std::int32_t lower_bound, 
    std::int32_t upper_bound,
    Impl implementation
) {
    std::size_t correct_count = 0;
    
    for (std::size_t i = 0; i < num_runs; ++i) {
        correct_count += (
            validate_range_of_matrices<4,   32> (lower_bound, upper_bound, implementation) +
            validate_range_of_matrices<36,  128>(lower_bound, upper_bound, implementation) +
            validate_range_of_matrices<132, 256>(lower_bound, upper_bound, implementation)
        );
    }

    return correct_count;
}

int main(int argsc, char** argsv) {
    if (argsc <= 3) {
        std::println("Specify [num runs] [rand lower bound] [rand upper bound]");
        return 0;
    }
    
    const std::size_t  num_runs    = std::stol(argsv[1]);
    const std::int32_t lower_bound = std::stoi(argsv[2]);
    const std::int32_t upper_bound = std::stoi(argsv[3]);

    assert(lower_bound < upper_bound && "Invalid bound. Lower bound < upper bound");

    const std::size_t ideal_correctness = (256 / 4) * num_runs;

    auto methods = std::to_array<std::pair<Impl, std::string_view>>({
        {Impl::TRANSPOSED, "Transposed"},
        {Impl::TILING, "Tiling"}
    });

    for (const auto& [implementation, name]: methods) {
        auto correct_count = validate_implementation(num_runs, lower_bound, upper_bound, implementation);
        double score = static_cast<double>(correct_count) 
                     / static_cast<double>(ideal_correctness) 
                     * 100.0;
        std::println("{} : {}/{} [{:.2f}%]", name, correct_count, ideal_correctness, score);
     }

    return 0;

}
