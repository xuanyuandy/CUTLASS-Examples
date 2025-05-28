cmake_minimum_required(VERSION 3.28)

project(CUTE-VECTOR-COPY-TEST VERSION 0.0.1 LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)

find_path(CUTE_VECTOR_COPY_INCLUDE cute_vector_copy.hpp HINTS ${CMAKE_SOURCE_DIR}/examples/cute_vector_copy/)

add_executable(test_vector_copy test_vector_copy.cu)
target_include_directories(test_vector_copy PRIVATE ${CUTE_VECTOR_COPY_INCLUDE})
target_link_libraries(test_vector_copy GTest::gtest GTest::gtest_main ${CUTE_VECTOR_COPY})
set_target_properties(test_vector_copy PROPERTIES CUDA_ARCHITECTURES native)

add_test(TestAllVectorCopy test_vector_copy)

add_executable(profile_vector_copy profile_vector_copy.cu)
target_include_directories(profile_vector_copy PRIVATE ${CUTE_VECTOR_COPY_INCLUDE})
target_link_libraries(profile_vector_copy GTest::gtest GTest::gtest_main ${CUTE_VECTOR_COPY})
set_target_properties(profile_vector_copy PROPERTIES CUDA_ARCHITECTURES native)

add_test(ProfileAllVectorCopy profile_vector_copy)

add_executable(test_vector_copy_vectorized test_vector_copy_vectorized.cu)
target_include_directories(test_vector_copy_vectorized PRIVATE ${CUTE_VECTOR_COPY_INCLUDE})
target_link_libraries(test_vector_copy_vectorized GTest::gtest GTest::gtest_main ${CUTE_VECTOR_COPY})
set_target_properties(test_vector_copy_vectorized PROPERTIES CUDA_ARCHITECTURES native)

add_test(TestAllVectorCopyVectorized test_vector_copy_vectorized)

add_executable(profile_vector_copy_vectorized profile_vector_copy_vectorized.cu)
target_include_directories(profile_vector_copy_vectorized PRIVATE ${CUTE_VECTOR_COPY_INCLUDE})
target_link_libraries(profile_vector_copy_vectorized GTest::gtest GTest::gtest_main ${CUTE_VECTOR_COPY})
set_target_properties(profile_vector_copy_vectorized PROPERTIES CUDA_ARCHITECTURES native)

add_test(ProfileAllVectorCopyVectorized profile_vector_copy_vectorized)
