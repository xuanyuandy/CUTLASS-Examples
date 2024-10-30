cmake_minimum_required(VERSION 3.28)

project(CUTE-TILED-COPY VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
add_executable(${PROJECT_NAME} tiled_copy.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTE_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${PROJECT_NAME} PRIVATE --expt-relaxed-constexpr)
