cmake_minimum_required(VERSION 3.23)
project(GEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -march=native)

include_directories(include)

add_executable(GEMM src/main.cpp)

# ---------- Tests ----------
enable_testing()

add_executable(gemm_tests tests/test_mat.cpp)
target_link_libraries(gemm_tests PRIVATE)

add_test(NAME GEMM.Tests COMMAND gemm_tests)

add_library(gemm_tests_constexpr OBJECT tests/test_mat_constexpr.cpp)

