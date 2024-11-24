#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_vector_copy.hpp"

static auto const LAUNCH_VECTOR_COPY_FLOAT{launch_vector_copy_vectorized<float>};

static auto const POWER_OF_TWO_VALUES{::testing::Values(1 << 28)};

TEST_P(TestVectorCopyFloat, TestVectorCopyFloat)
{
    MeasurePerformance(LAUNCH_VECTOR_COPY_FLOAT);
}

INSTANTIATE_TEST_SUITE_P(TestVectorCopyPowerOfTwo, TestVectorCopyFloat,
                         POWER_OF_TWO_VALUES);
