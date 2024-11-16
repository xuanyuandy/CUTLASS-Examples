#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_transpose.hpp"

static const auto LAUNCH_TRANSPOSE_FLOAT{
    launch_transpose_naive_coalesced_read<float>};

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(16384)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(16384)};

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    // RunTest(LAUNCH_TRANSPOSE_FLOAT); // Skip CPU test because large problems
    // on CPU are slow.
    MeasurePerformance(LAUNCH_TRANSPOSE_FLOAT);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
