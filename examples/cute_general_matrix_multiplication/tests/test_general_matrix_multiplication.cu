#include <gtest/gtest.h>

#include "test_utils.hpp"

#include "cute_general_matrix_multiplication.hpp"

static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_FLOAT{
    launch_sgemm_1<float, float, float, float, float>};
static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_DOUBLE{
    launch_sgemm_1<double, double, double, double, double>};
static auto const LAUNCH_GENERAL_MATRIX_MULTIPLICATION_HALF{
    launch_sgemm_1<cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
                   float>};

static auto const TRANS_A_VALUES{::testing::Values('N')};
static auto const TRANS_B_VALUES{::testing::Values('T')};
static auto const M_VALUES{::testing::Values(256)};
static auto const N_VALUES{::testing::Values(256)};
static auto const K_VALUES{::testing::Values(256)};
static auto const LDA_VALUES{::testing::Values(256)};
static auto const LDB_VALUES{::testing::Values(256)};
static auto const LDC_VALUES{::testing::Values(256)};

TEST_P(TestGeneralMatrixMultiplicationFloat,
       TestGeneralMatrixMultiplicationFloat)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_FLOAT);
}

TEST_P(TestGeneralMatrixMultiplicationDouble,
       TestGeneralMatrixMultiplicationDouble)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_DOUBLE);
}

TEST_P(TestGeneralMatrixMultiplicationHalf, TestGeneralMatrixMultiplicationHalf)
{
    RunTest(LAUNCH_GENERAL_MATRIX_MULTIPLICATION_HALF);
}

INSTANTIATE_TEST_SUITE_P(TestGeneralMatrixMultiplicationLimited,
                         TestGeneralMatrixMultiplicationFloat,
                         ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES,
                                            M_VALUES, N_VALUES, K_VALUES,
                                            LDA_VALUES, LDB_VALUES,
                                            LDC_VALUES));
INSTANTIATE_TEST_SUITE_P(TestGeneralMatrixMultiplicationLimited,
                         TestGeneralMatrixMultiplicationDouble,
                         ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES,
                                            M_VALUES, N_VALUES, K_VALUES,
                                            LDA_VALUES, LDB_VALUES,
                                            LDC_VALUES));
INSTANTIATE_TEST_SUITE_P(
    TestGeneralMatrixMultiplicationLimited, TestGeneralMatrixMultiplicationHalf,
    ::testing::Combine(TRANS_A_VALUES, TRANS_B_VALUES, M_VALUES, N_VALUES,
                       K_VALUES, LDA_VALUES, LDB_VALUES, LDC_VALUES));
