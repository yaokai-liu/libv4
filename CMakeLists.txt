cmake_minimum_required(VERSION 3.28)
project(vector4 C)

set(CMAKE_C_STANDARD 23)
add_compile_options("-msse4" "-mavx2" "-O3")

include_directories(.)

add_library(vector4 STATIC vector4.c)
add_executable(test test/main.c)
target_link_libraries(test PRIVATE vector4)
