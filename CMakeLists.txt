cmake_minimum_required(VERSION 3.23)
project(GEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -march=native)

include_directories(include)

add_executable(GEMM src/main.cpp)
