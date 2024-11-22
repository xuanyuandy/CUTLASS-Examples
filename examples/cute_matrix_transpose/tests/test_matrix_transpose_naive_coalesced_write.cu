#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_matrix_transpose.hpp"

static auto const LAUNCH_MATRIX_TRANSPOSE_FLOAT{
    launch_matrix_transpose_naive_coalesced_write<float>};
static auto const LAUNCH_MATRIX_TRANSPOSE_DOUBLE{
    launch_matrix_transpose_naive_coalesced_write<double>};

static auto const M_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};
static auto const N_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};

static auto const M_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};
static auto const N_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};

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
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeDouble,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeFloat,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeDouble,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));