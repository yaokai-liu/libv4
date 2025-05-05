cmake_minimum_required(VERSION 3.28)
project(xglm C)

set(CMAKE_C_STANDARD 23)
add_compile_options("-msse4" "-mavx2" "-O3")

include_directories(src)

aux_source_directory(src SRC)
add_library(xglm STATIC ${SRC})

if (XGLM_BUILD_DEBUG)
    add_executable(debug test/debug.c)
    target_link_libraries(debug PRIVATE xglm)
endif ()