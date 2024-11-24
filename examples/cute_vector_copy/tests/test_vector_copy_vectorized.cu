#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_vector_copy.hpp"

static auto const LAUNCH_VECTOR_COPY_FLOAT{
    launch_vector_copy_vectorized<float>};
static auto const LAUNCH_VECTOR_COPY_DOUBLE{
    launch_vector_copy_vectorized<double>};

static auto const MULTIPLE_OF_FOUR_VALUES{
    ::testing::Values(4, 4 * 541, 4 * 1123, 4 * 20491)};
static auto const MULTIPLE_OF_TWO_VALUES{
    ::testing::Values(2, 2 * 541, 2 * 1123, 2 * 20491)};

TEST_P(TestVectorCopyFloat, TestVectorCopyFloat)
{
    RunTest(LAUNCH_VECTOR_COPY_FLOAT);
}

TEST_P(TestVectorCopyDouble, TestVectorCopyDouble)
{
    RunTest(LAUNCH_VECTOR_COPY_DOUBLE);
}

INSTANTIATE_TEST_SUITE_P(TestVectorCopy, TestVectorCopyFloat,
                         MULTIPLE_OF_FOUR_VALUES);
INSTANTIATE_TEST_SUITE_P(TestVectorCopy, TestVectorCopyDouble,
                         MULTIPLE_OF_TWO_VALUES);
