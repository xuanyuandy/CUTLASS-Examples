cmake_minimum_required(VERSION 3.28)

project(CUTE-TRANSPOSE-NAIVE-TEST VERSION 0.0.1 LANGUAGES CXX CUDA)

find_package(GTest REQUIRED)
# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

find_path(CUTE_TRANSPOSE_INCLUDE cute_transpose_naive.hpp HINTS ${CMAKE_SOURCE_DIR}/examples/cute_transpose_naive)

add_executable(test_matrix_transpose test_matrix_transpose.cu)
# include the directories for cute_transpose_naive.hpp
target_include_directories(test_matrix_transpose PRIVATE ${CUTE_TRANSPOSE_INCLUDE})
target_link_libraries(test_matrix_transpose GTest::gtest GTest::gtest_main ${CUTE_TRANSPOSE_NAIVE})
set_target_properties(test_matrix_transpose PROPERTIES CUDA_ARCHITECTURES native)

add_test(AllMatrixTransposeTests test_matrix_transpose)
