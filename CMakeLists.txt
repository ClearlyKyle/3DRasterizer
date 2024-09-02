cmake_minimum_required(VERSION 3.10)

project(main)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_C_STANDARD 11) # Set the desired C standard (e.g., C11)

if(MSVC)
    add_compile_options(/W4) # Set warning level to 4
    add_compile_options(/std:c11 /sdl /guard:cf /analyze)
    add_compile_options(/openmp:llvm)
endif()

find_package(SDL2 CONFIG REQUIRED)

include_directories(src/matematika)
include_directories(libs)

set(RESTER_SOURCE
    "src/main.c"
    "src/Renderer.h"
    "src/Renderer.c"
    "src/lights.h"
    "src/lights.c"
    "src/obj.h"
    "src/obj.c"
    "src/app.h"
    "src/app.c"
    "src/textures.h"
    "src/textures.c"
)

add_executable(${CMAKE_PROJECT_NAME} ${RESTER_SOURCE})

# SDL2::SDL2main needs to come first to compile with GCC 10.3.0
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE SDL2::SDL2main SDL2::SDL2)
