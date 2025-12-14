cmake_minimum_required(VERSION 3.23)
project(GEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wno-maybe-uninitialized)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release (-O3)")
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

add_library(gemm INTERFACE)
target_include_directories(gemm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ffile-prefix-map=${CMAKE_SOURCE_DIR}=${CMAKE_SOURCE_DIR}")
add_executable(gemm_benchmark apps/gemm_benchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE 
    gemm
    benchmark::benchmark 
    benchmark::benchmark_main
)

add_executable(validate_correctness apps/validate_correctness.cpp)
target_link_libraries(validate_correctness PRIVATE gemm)

add_executable(perf_driver apps/perf_driver.cpp)
target_link_libraries(perf_driver PRIVATE gemm)

# ---------- Tests ----------
enable_testing()

add_executable(gemm_tests tests/test_mat.cpp)
target_link_libraries(gemm_tests PRIVATE gemm)
add_test(NAME GEMM.Tests COMMAND gemm_tests)

add_library(gemm_tests_constexpr OBJECT tests/test_mat_constexpr.cpp)

