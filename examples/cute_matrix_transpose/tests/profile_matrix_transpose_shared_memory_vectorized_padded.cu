#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_matrix_transpose.hpp"

static auto const LAUNCH_MATRIX_TRANSPOSE_FLOAT{
    launch_matrix_transpose_shared_memory_vectorized_padded<float>};

static auto const M_POWER_OF_TWO_VALUES{::testing::Values(16384)};
static auto const N_POWER_OF_TWO_VALUES{::testing::Values(16384)};

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    // RunTest(LAUNCH_MATRIX_TRANSPOSE_FLOAT); // Skip CPU test
    // because large problems on CPU are slow.
    MeasurePerformance(LAUNCH_MATRIX_TRANSPOSE_FLOAT);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
