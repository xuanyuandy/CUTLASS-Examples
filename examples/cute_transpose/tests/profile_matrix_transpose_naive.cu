#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_transpose.hpp"

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(16384)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(16384)};

TEST_P(TestMatrixTransposeInt, TestMatrixTransposeInt)
{
    // RunTest(launch_transpose_naive); // Skip CPU test because large problems
    // on CPU are slow.
    MeasurePerformance(launch_transpose_naive);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo, TestMatrixTransposeInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
