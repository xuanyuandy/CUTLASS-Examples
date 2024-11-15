#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_transpose.hpp"

static const auto LAUNCH_TRANSPOSE_INT{
    launch_transpose_shared_memory_bank_conflict_write<int>};

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(16384)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(16384)};

TEST_P(TestMatrixTransposeInt, TestMatrixTransposeInt)
{
    // RunTest(LAUNCH_TRANSPOSE_INT); // Skip CPU test
    // because large problems on CPU are slow.
    MeasurePerformance(LAUNCH_TRANSPOSE_INT);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo, TestMatrixTransposeInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
