#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_vector_copy.hpp"

static auto const LAUNCH_VECTOR_COPY_FLOAT{launch_vector_copy<float>};
static auto const LAUNCH_VECTOR_COPY_DOUBLE{launch_vector_copy<double>};

static auto const PRIME_VALUES{
    ::testing::Values(2, 17, 83, 163, 257, 3659, 5821)};

static auto const POWER_OF_TWO_VALUES{
    ::testing::Values(1, 16, 256, 1024, 32768)};

TEST_P(TestVectorCopyFloat, TestVectorCopyFloat)
{
    RunTest(LAUNCH_VECTOR_COPY_FLOAT);
}

TEST_P(TestVectorCopyDouble, TestVectorCopyDouble)
{
    RunTest(LAUNCH_VECTOR_COPY_DOUBLE);
}

INSTANTIATE_TEST_SUITE_P(TestVectorCopyPrime, TestVectorCopyFloat,
                         PRIME_VALUES);
INSTANTIATE_TEST_SUITE_P(TestVectorCopyPrime, TestVectorCopyDouble,
                         PRIME_VALUES);
INSTANTIATE_TEST_SUITE_P(TestVectorCopyPowerOfTwo, TestVectorCopyFloat,
                         POWER_OF_TWO_VALUES);
INSTANTIATE_TEST_SUITE_P(TestVectorCopyPowerOfTwo, TestVectorCopyDouble,
                         POWER_OF_TWO_VALUES);
