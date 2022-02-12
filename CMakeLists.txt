cmake_minimum_required(VERSION 3.10)

project(main)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(SDL2 CONFIG REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.c vector.h vector.c Renderer.c Renderer.h)
add_executable(VECTOR_TEST vector_test.c vector.h vector.c)

# SDL2::SDL2main needs to come first to compile with GCC 10.3.0
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE SDL2::SDL2main SDL2::SDL2)
