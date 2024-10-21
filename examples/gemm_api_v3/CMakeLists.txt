cmake_minimum_required(VERSION 3.28)

project(CUTLASS-GEMM-API-V3 VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUTLASS include directories
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS /opt/cutlass/include)
find_path(CUTE_INCLUDE_DIR cute/tensor.hpp HINTS /opt/cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS /opt/cutlass/tools/util/include)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
add_executable(${PROJECT_NAME} main.cu)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUTLASS_INCLUDE_DIR} ${CUTE_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
