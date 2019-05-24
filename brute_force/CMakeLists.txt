# An Example Script for CUDA with CMake

# CUDA support was introduced in version 3.8 and 
# will not work if there is a version mismatch
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_HOST_COMPILER $ENV{CU_CCBIN} CACHE PATH "Cuda host compiler dir")

# Set up project with C and CUDA language support
project(Final_Project LANGUAGES C CXX CUDA)

# Set compiler flags
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE, Release)

add_executable(collide main.cu ../tiny_obj_loader.h)
