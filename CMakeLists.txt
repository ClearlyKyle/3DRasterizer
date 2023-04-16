cmake_minimum_required(VERSION 3.10)

project(main)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(MSVC)
    # Set warning level to 4
    add_compile_options(/W4)
endif()

find_package(SDL2 CONFIG REQUIRED)

set(RESTER_SOURCE
    "main.c"
    "vector.h"
    "vector.c"
    "Renderer.h"
    "Renderer.c"
    "lights.h"
    "lights.c"
    "textures.h"
    "textures.c"
    "AABB.h"
    "ObjLoader.h"
    "timer.h"
)

add_executable(${CMAKE_PROJECT_NAME} ${RESTER_SOURCE})
add_executable(VECTOR_TEST vector_test.c vector.h vector.c lights.h lights.c)
add_executable(OBJECT_TEST ObjLoader_test.c ObjLoader.h)

# SDL2::SDL2main needs to come first to compile with GCC 10.3.0
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE SDL2::SDL2main SDL2::SDL2)
