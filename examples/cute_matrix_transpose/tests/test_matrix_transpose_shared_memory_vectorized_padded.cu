#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_matrix_transpose.hpp"

static auto const LAUNCH_MATRIX_TRANSPOSE_FLOAT{
    launch_matrix_transpose_shared_memory_vectorized_padded<float>};
static auto const LAUNCH_MATRIX_TRANSPOSE_DOUBLE{
    launch_matrix_transpose_shared_memory_vectorized_padded<double>};

static auto const M_MULTIPLE_OF_FOUR_VALUES{
    ::testing::Values(2 * 4, 17 * 4, 83 * 4, 163 * 4, 257 * 4)};
static auto const N_MULTIPLE_OF_FOUR_VALUES{
    ::testing::Values(2 * 4, 17 * 4, 83 * 4, 163 * 4, 257 * 4)};

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    RunTest(LAUNCH_MATRIX_TRANSPOSE_FLOAT);
}

TEST_P(TestMatrixTransposeDouble, TestMatrixTransposeDouble)
{
    RunTest(LAUNCH_MATRIX_TRANSPOSE_DOUBLE);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_MULTIPLE_OF_FOUR_VALUES,
                                            N_MULTIPLE_OF_FOUR_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeDouble,
                         ::testing::Combine(M_MULTIPLE_OF_FOUR_VALUES,
                                            N_MULTIPLE_OF_FOUR_VALUES));
