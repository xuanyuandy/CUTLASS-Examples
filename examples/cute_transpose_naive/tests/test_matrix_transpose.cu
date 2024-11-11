#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cute_transpose_naive.hpp"

#include "test_utils.hpp"

static const auto M_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};
static const auto N_PRIME_VALUES{::testing::Values(2, 17, 83, 163, 257)};

static const auto M_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};
static const auto N_POWER_OF_TWO_VALUES{::testing::Values(1, 16, 256, 1024)};

template <typename T>
class TestMatrixTranspose
    : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>>
{
protected:
    void SetUp() override
    {
        // Create CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));

        // Get paramater.
        m_M = std::get<0>(GetParam());
        m_N = std::get<1>(GetParam());

        // Use thrust to create the host and device vectors.
        m_h_src = thrust::host_vector<T>(m_M * m_N);
        m_h_dst = thrust::host_vector<T>(m_N * m_M);
        m_h_dst_ref = thrust::host_vector<T>(m_N * m_M);

        m_d_src = thrust::device_vector<T>(m_M * m_N);
        m_d_dst = thrust::device_vector<T>(m_N * m_M);

        // Initialize the host vectors.
        initialize(m_h_src.data(), m_h_src.size());
        transpose(m_h_src.data(), m_h_dst_ref.data(), m_M, m_N);

        // Copy the host vectors to the device vectors.
        m_d_src = m_h_src;
    }

    void TearDown() override
    {
        // Destroy CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamDestroy(m_stream));
    }

    void RunTest()
    {
        // Launch the kernel.
        CHECK_CUDA_ERROR(launch_transpose_naive(
            thrust::raw_pointer_cast(m_d_src.data()),
            thrust::raw_pointer_cast(m_d_dst.data()), m_M, m_N, m_stream));
        
        // Synchronize the stream.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

        // Copy the data from device to host.
        m_h_dst = m_d_dst;

        // Compare the data.
        ASSERT_TRUE(
            compare(m_h_dst.data(), m_h_dst_ref.data(), m_h_dst.size()));
    }

    unsigned int m_M;
    unsigned int m_N;

    cudaStream_t m_stream;

    thrust::host_vector<T> m_h_src;
    thrust::host_vector<T> m_h_dst;
    thrust::host_vector<T> m_h_dst_ref;

    thrust::device_vector<T> m_d_src;
    thrust::device_vector<T> m_d_dst;
};

class TestMatrixTransposeInt : public TestMatrixTranspose<int>
{
};

class TestMatrixTransposeUnsignedInt : public TestMatrixTranspose<unsigned int>
{
};

class TestMatrixTransposeFloat : public TestMatrixTranspose<float>
{
};

class TestMatrixTransposeDouble : public TestMatrixTranspose<double>
{
};

TEST_P(TestMatrixTransposeInt, TestMatrixTransposeInt) { RunTest(); }

TEST_P(TestMatrixTransposeUnsignedInt, TestMatrixTransposeUnsignedInt)
{
    RunTest();
}

TEST_P(TestMatrixTransposeFloat, TestMatrixTransposeFloat) { RunTest(); }

TEST_P(TestMatrixTransposeDouble, TestMatrixTransposeDouble) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo, TestMatrixTransposeInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeUnsignedInt,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeFloat,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePowerOfTwo,
                         TestMatrixTransposeDouble,
                         ::testing::Combine(M_POWER_OF_TWO_VALUES,
                                            N_POWER_OF_TWO_VALUES));

INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeInt,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime,
                         TestMatrixTransposeUnsignedInt,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeFloat,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));
INSTANTIATE_TEST_SUITE_P(TestMatrixTransposePrime, TestMatrixTransposeDouble,
                         ::testing::Combine(M_PRIME_VALUES, N_PRIME_VALUES));