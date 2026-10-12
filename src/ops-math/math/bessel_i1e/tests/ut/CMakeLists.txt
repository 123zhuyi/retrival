cmake_minimum_required(VERSION 3.16.0)
project(bessel_i1e_ut CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_bessel_i1e_ut test_bessel_i1e_ut.cpp)
target_compile_options(test_bessel_i1e_ut PRIVATE -Wall -Wextra -O2)
