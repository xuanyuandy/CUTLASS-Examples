cmake_minimum_required(VERSION 3.28)

project(CUTE-TRANSPOSE VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_TRANSPOSE "CUTE_TRANSPOSE")
add_executable(${CUTE_TRANSPOSE} cute_transpose.cu)
target_include_directories(${CUTE_TRANSPOSE} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_TRANSPOSE} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_TRANSPOSE} PRIVATE --expt-relaxed-constexpr)
