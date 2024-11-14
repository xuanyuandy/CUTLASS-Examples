#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_transpose.hpp"

static const auto LAUNCH_TRANSPOSE_INT{
    launch_transpose_shared_memory_padded<int>};
static const auto LAUNCH_TRANSPOSE_UINT{
    launch_transpose_shared_memory_padded<unsigned int>};
static const auto LAUNCH_TRANSPOSE_FLOAT{
    launch_transpose_shared_memory_padded<float>};
static const auto LAUNCH_TRANSPOSE_DOUBLE{
    launch_transpose_shared_memory_padded<double>};

static const auto M_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};
static const auto N_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};

TEST_P(TestMatrixTransposeInt, TestMatrixTransposeInt)
{
    RunTest(LAUNCH_TRANSPOSE_INT);
}

TEST_P(TestMatrixTransposeUnsignedInt, TestMatrixTransposeUnsignedInt)
{
    RunTest(LAUNCH_TRANSPOSE_UINT);
}

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    RunTest(LAUNCH_TRANSPOSE_FLOAT);
}

TEST_P(TestMatrixTransposeDouble, TestMatrixTransposeDouble)
{
    RunTest(LAUNCH_TRANSPOSE_DOUBLE);
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