#include "mat.hpp"

#include <algorithm>
#include <print>
#include <string>
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
    return a.multiply(b, Impl::NAIVE) == a.multiply(b, implementation);
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

    auto transposed_correct = validate_implementation(num_runs, lower_bound, upper_bound, Impl::TRANSPOSED);
    auto tiling_correct     = validate_implementation(num_runs, lower_bound, upper_bound, Impl::TILING);

    double transposed_score = static_cast<double>(transposed_correct) 
                            / static_cast<double>(ideal_correctness) 
                            * 100.0;

    double tiling_score = static_cast<double>(tiling_correct) 
                        / static_cast<double>(ideal_correctness) 
                        * 100.0;

    std::println("transposed : {}/{} [{:.2f}%]", transposed_correct, ideal_correctness, transposed_score);
    std::println("tiling     : {}/{} [{:.2f}%]", tiling_correct,     ideal_correctness, tiling_score);

    return 0;

}
