cmake_minimum_required(VERSION 3.28)

project(CUTE-GENERAL-MATRIX-MULTIPLICATION VERSION 0.0.1 LANGUAGES CXX CUDA)

# Build specialized kernels that performs no bounds check and only supports a few problem shapes.
# This will only be used for checking the highest performance possible for general matrix multiplication.
# The comparison to cuBLAS will not be unfair because cuBLAS consists of a large number of kernels each of which is specialized for a specific problem shape.
option(NO_BOUNDS_CHECK "CUDA kernel performs no bounds check" OFF)
if (NO_BOUNDS_CHECK)
    add_definitions(-DNO_BOUNDS_CHECK)
endif()

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_GENERAL_MATRIX_MULTIPLICATION cute_general_matrix_multiplication)
add_library(
    ${CUTE_GENERAL_MATRIX_MULTIPLICATION}
    SHARED
    cute_general_matrix_multiplication_naive.cu
    cute_general_matrix_multiplication_naive_gmem_tiled_copy_tiled_mma.cu
    cute_general_matrix_multiplication_naive_gmem_tiled_copy_tiled_mma_sm70_pipeline.cu
    cute_general_matrix_multiplication_tensor_core_gmem_tiled_copy_tiled_mma.cu
    cute_general_matrix_multiplication_tensor_core_gmem_tiled_copy_smem_tiled_copy_tiled_mma.cu
    cute_general_matrix_multiplication_tensor_core_gmem_tiled_copy_tiled_mma_sm80_pipeline.cu
)
target_include_directories(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PRIVATE --expt-relaxed-constexpr)
# target_compile_definitions(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PRIVATE NO_BOUNDS_CHECK=1)

add_subdirectory(tests)