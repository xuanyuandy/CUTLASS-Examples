#include <gtest/gtest.h>

#include "test_utils.hpp"

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(16384)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(16384)};

TEST_P(TestMatrixTransposeInt, TestMatrixTransposeInt)
{
    // RunTest(); // Skip CPU test because large problems on CPU are slow.
    MeasurePerformance();
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo, TestMatrixTransposeInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
