#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_vector_copy.hpp"

static auto const LAUNCH_VECTOR_COPY_FLOAT{launch_vector_copy<float>};
static auto const LAUNCH_VECTOR_COPY_DOUBLE{launch_vector_copy<double>};

// static auto const M_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};
// static auto const N_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};

static auto const SIZE_POWER_OF_TWO_VALUES{::testing::Values(4093)};

TEST_P(TestVectorCopyFloat, TestVectorCopyFloat)
{
    RunTest(LAUNCH_VECTOR_COPY_FLOAT);
}

TEST_P(TestVectorCopyDouble, TestVectorCopyDouble)
{
    RunTest(LAUNCH_VECTOR_COPY_DOUBLE);
}

INSTANTIATE_TEST_SUITE_P(TestVectorCopyPowerOfTwo, TestVectorCopyFloat,
                         SIZE_POWER_OF_TWO_VALUES);
INSTANTIATE_TEST_SUITE_P(TestVectorCopyPowerOfTwo, TestVectorCopyDouble,
                         SIZE_POWER_OF_TWO_VALUES);

// INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeFloat,
//                          ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
// INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeDouble,
//                          ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));