cmake_minimum_required(VERSION 3.28)
project(vector4 C)

set(CMAKE_C_STANDARD 23)
add_compile_options("-msse4" "-mavx2" "-O3")

include_directories(src)

aux_source_directory(src SRC)
add_library(vector4 STATIC ${SRC})
add_executable(test test/main.c)
target_link_libraries(test PRIVATE vector4)
