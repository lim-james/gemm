cmake_minimum_required(VERSION 3.23)
project(GEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-g -fno-omit-frame-pointer)
add_compile_options(-O3)
# add_compile_options(-O3 -march=native)

add_library(gemm INTERFACE)
target_include_directories(gemm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(benchmark apps/benchmark.cpp)
target_link_libraries(benchmark PRIVATE gemm)

add_executable(validate_correctness apps/validate_correctness.cpp)
target_link_libraries(validate_correctness PRIVATE gemm)

add_executable(perf_driver apps/perf_driver.cpp)
target_link_libraries(perf_driver PRIVATE gemm)

# ---------- Tests ----------
enable_testing()

add_executable(gemm_tests tests/test_mat.cpp)
target_link_libraries(gemm_tests PRIVATE)
add_test(NAME GEMM.Tests COMMAND gemm_tests)

add_library(gemm_tests_constexpr OBJECT tests/test_mat_constexpr.cpp)

