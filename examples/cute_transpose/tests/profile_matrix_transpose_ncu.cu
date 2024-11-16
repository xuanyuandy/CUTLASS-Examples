#include <vector>

#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_transpose.hpp"

static std::vector<cudaError_t (*)(float const*, float*, unsigned int,
                                   unsigned int, cudaStream_t)> const
    transpose_functions{
        launch_transpose_naive_coalesced_read<float>,
        launch_transpose_naive_coalesced_write<float>,
        launch_transpose_shared_memory_bank_conflict_read<float>,
        launch_transpose_shared_memory_bank_conflict_write<float>,
        launch_transpose_shared_memory_padded<float>,
        launch_transpose_shared_memory_swizzled<float>};

static auto const M_POWER_OF_TWO_VALUES{::testing::Values(16384)};
static auto const N_POWER_OF_TWO_VALUES{::testing::Values(16384)};

static unsigned int const NUM_REPEATS{1U};
static unsigned int const NUM_WARMUPS{0U};

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat)
{
    for (auto const& transpose_function : transpose_functions)
    {
        std::cout << "NumRepeats: " << NUM_REPEATS
                  << " NumWarmups: " << NUM_WARMUPS << std::endl;
        MeasurePerformance(transpose_function, NUM_REPEATS, NUM_WARMUPS);
    }
}

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
