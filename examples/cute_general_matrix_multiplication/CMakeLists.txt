cmake_minimum_required(VERSION 3.28)

project(CUTE-GENERAL-MATRIX-MULTIPLICATION VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_GENERAL_MATRIX_MULTIPLICATION cute_general_matrix_multiplication)
add_library(
    ${CUTE_GENERAL_MATRIX_MULTIPLICATION}
    SHARED
    cute_general_matrix_multiplication_naive.cu
    cute_general_matrix_multiplication_naive_tiled_copy_tiled_mma.cu
    cute_general_matrix_multiplication_tensor_core_tiled_copy_tiled_mma.cu
)
target_include_directories(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_GENERAL_MATRIX_MULTIPLICATION} PRIVATE --expt-relaxed-constexpr)

add_subdirectory(tests)