cmake_minimum_required(VERSION 3.28)

project(CUTE-TRANSPOSE-NAIVE VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_TRANSPOSE_NAIVE cute_transpose_naive)
add_library(${CUTE_TRANSPOSE_NAIVE} SHARED cute_transpose_naive.cu)
target_include_directories(${CUTE_TRANSPOSE_NAIVE} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_TRANSPOSE_NAIVE} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_TRANSPOSE_NAIVE} PRIVATE --expt-relaxed-constexpr)

add_subdirectory(tests)