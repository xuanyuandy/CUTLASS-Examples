#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_transpose.hpp"

static const auto M_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};
static const auto N_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};

TEST_P(TestMatrixTransposeInt, TestMatrixTransposeInt)
{
    RunTest(launch_transpose_naive);
}

TEST_P(TestMatrixTransposeUnsignedInt, TestMatrixTransposeUnsignedInt)
{
    RunTest(launch_transpose_naive);
}

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    RunTest(launch_transpose_naive);
}

TEST_P(TestMatrixTransposeDouble, TestMatrixTransposeDouble)
{
    RunTest(launch_transpose_naive);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo, TestMatrixTransposeInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeUnsignedInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeDouble,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeInt,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime,
                         TestMatrixTransposeUnsignedInt,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeFloat,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeDouble,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));