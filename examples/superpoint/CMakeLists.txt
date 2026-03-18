# SuperPoint C++ inference example for Axelera Metis M2 (iMX8MP)
cmake_minimum_required(VERSION 3.12)
project(superpoint_inference LANGUAGES CXX VERSION 0.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4   REQUIRED opencv4)
pkg_check_modules(axstreamer REQUIRED axstreamer)

add_executable(superpoint_inference
    superpoint_main.cpp
    superpoint_postprocess.cpp
)

target_compile_options(superpoint_inference PRIVATE
    -Wall
    -Wextra
    -O3
    -march=armv8-a+simd        # enable NEON auto-vectorisation on Cortex-A55
    -ffast-math                # allow FP reassociation (safe for our softmax)
    -Wno-sign-compare
    -Wno-unused-variable
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

target_include_directories(superpoint_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${axruntime_INCLUDE_DIRS}/include
    ${opencv4_INCLUDE_DIRS}
    ${axstreamer_INCLUDE_DIRS}
)

target_link_libraries(superpoint_inference PUBLIC
    ${axruntime_LIBRARIES}
    ${opencv4_LIBRARIES}
    ${axstreamer_LIBRARIES}
)

target_link_directories(superpoint_inference PUBLIC
    ${axruntime_LIBRARY_DIRS}
    ${opencv4_LIBRARY_DIRS}
    ${axstreamer_LIBRARY_DIRS}
)

install(TARGETS superpoint_inference RUNTIME DESTINATION bin)
