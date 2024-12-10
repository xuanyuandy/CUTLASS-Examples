#include <iostream>
#include <random>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cutlass/half.h>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// The original matrix is column major.
// coordinate (i, j) is from the transposed matrix if there is any.
int coordinate_to_index(int i, int j, int ld, bool trans)
{
    if (trans)
    {
        return j + i * ld;
    }
    else
    {
        return i + j * ld;
    }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transa, char transb, int m, int n, int k, Alpha alpha,
          TA const* A, int lda, TB const* B, int ldb, Beta beta, TC* C, int ldc)
{
    bool const transa_bool{transa == 'T' || transa == 't'};
    bool const transb_bool{transb == 'T' || transb == 't'};

    for (int i{0}; i < m; ++i)
    {
        for (int j{0}; j < n; ++j)
        {
            TC sum{0};
            for (int l{0}; l < k; ++l)
            {
                int const idx_a{
                    coordinate_to_index(i, l, lda, transa_bool)}; // A[i, l]
                int const idx_b{
                    coordinate_to_index(l, j, ldb, transb_bool)}; // B[l, j]
                TA const a{A[idx_a]};
                TB const b{B[idx_b]};
                sum += a * b;
            }
            int const idx_c{coordinate_to_index(i, j, ldc, false)}; // C[i, j]
            TC const c{C[idx_c]};
            C[idx_c] = alpha * sum + beta * c;
        }
    }
}

// Initialize a data array.
template <class T>
void initialize(T* data, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        data[i] = static_cast<T>(i);
    }
}

template <class T>
void random_initialize(T* data, unsigned int size, T a, T b,
                       std::default_random_engine& random_engine)
{
    std::uniform_int_distribution<int> distribution(static_cast<int>(a),
                                                    static_cast<int>(b));
    for (unsigned int i{0}; i < size; ++i)
    {
        data[i] = static_cast<T>(distribution(random_engine));
    }
}

// Print a data array and a reference array.
template <class T>
void print(T const* data, T const* ref, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        std::cout << i << " " << data[i] << " " << ref[i] << std::endl;
    }
}

// Compare a data array with a reference array.
template <class T>
bool compare(T const* data, T const* ref, unsigned int size)
{
    bool status{true};
    for (unsigned int i{0}; i < size; ++i)
    {
        if (data[i] != ref[i])
        {
            status = false;
        }
    }

    if (!status)
    {
        print<T>(data, ref, size);
    }

    return status;
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> const& bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 20,
                          unsigned int num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0U}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

template <class T>
float convert_latency_to_effective_bandwidth(float latency, unsigned int M,
                                             unsigned int N)
{
    size_t const size{M * N * sizeof(T) * 2};
    float const bandwidth{size / (latency / 1.0e3f) / (1 << 30)};
    return bandwidth;
}

template <class TA, class TB, class TC, class Alpha, class Beta>
class TestGeneralMatrixMultiplication
    : public ::testing::TestWithParam<
          std::tuple<char, char, int, int, int, int, int, int>>
{
protected:
    void SetUp() override
    {
        // Set random seed.
        m_seed = 0;
        m_random_engine = std::default_random_engine{m_seed};

        TA a_lower_bound{static_cast<TA>(-2.0)};
        TA a_upper_bound{static_cast<TA>(2.0)};
        TB b_lower_bound{static_cast<TB>(-2.0)};
        TB b_upper_bound{static_cast<TB>(2.0)};

        // Create CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));

        // Get parameter.
        m_transa = std::get<0>(GetParam());
        m_transb = std::get<1>(GetParam());

        // Check if the char is valid.
        ASSERT_TRUE(m_transa == 'N' || m_transa == 'n' || m_transa == 'T' ||
                    m_transa == 't');
        ASSERT_TRUE(m_transb == 'N' || m_transb == 'n' || m_transb == 'T' ||
                    m_transb == 't');

        m_m = std::get<2>(GetParam());
        m_n = std::get<3>(GetParam());
        m_k = std::get<4>(GetParam());

        m_lda = std::get<5>(GetParam());
        m_ldb = std::get<6>(GetParam());
        m_ldc = std::get<7>(GetParam());

        // Compute matrix size.
        // All matrices are column major.
        // op(A) is m x k.
        // op(B) is k x n.
        // C is m x n.
        int const num_ldas{m_transa == 'N' || m_transa == 'n' ? m_k : m_m};
        int const num_ldbs{m_transb == 'N' || m_transb == 'n' ? m_n : m_k};
        int const num_ldcs{m_n};
        int const size_a{num_ldas * m_lda};
        int const size_b{num_ldbs * m_ldb};
        int const size_c{num_ldcs * m_ldc};

        // Use thrust to create the host and device vectors.
        m_h_A = thrust::host_vector<TA>(size_a);
        m_h_B = thrust::host_vector<TB>(size_b);
        m_h_C = thrust::host_vector<TC>(size_c);
        m_h_C_ref = thrust::host_vector<TC>(size_c);

        m_d_A = thrust::device_vector<TA>(size_a);
        m_d_B = thrust::device_vector<TB>(size_b);
        m_d_C = thrust::device_vector<TC>(size_c);

        // Initialize coefficients.
        m_alpha = static_cast<Alpha>(1.0);
        m_beta = static_cast<Beta>(0.0);

        // Initialize the host vectors.
        random_initialize(m_h_A.data(), m_h_A.size(), a_lower_bound,
                          a_upper_bound, m_random_engine);
        random_initialize(m_h_B.data(), m_h_B.size(), b_lower_bound,
                          b_upper_bound, m_random_engine);

        // Compute the reference result.
        gemm(m_transa, m_transb, m_m, m_n, m_k, m_alpha, m_h_A.data(), m_lda,
             m_h_B.data(), m_ldb, m_beta, m_h_C_ref.data(), m_ldc);

        // Copy the host vectors to the device vectors.
        m_d_A = m_h_A;
        m_d_B = m_h_B;
    }

    void TearDown() override
    {
        // Destroy CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamDestroy(m_stream));
    }

    void RunTest(cudaError_t (*launch_gemm)(char, char, int, int, int, Alpha,
                                            TA const*, int, TB const*, int,
                                            Beta, TC*, int, cudaStream_t))
    {
        // Launch the kernel.
        CHECK_CUDA_ERROR(launch_gemm(
            m_transa, m_transb, m_m, m_n, m_k, m_alpha,
            thrust::raw_pointer_cast(m_d_A.data()), m_lda,
            thrust::raw_pointer_cast(m_d_B.data()), m_ldb, m_beta,
            thrust::raw_pointer_cast(m_d_C.data()), m_ldc, m_stream));

        // Synchronize the stream.
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));

        // Copy the data from device to host.
        m_h_C = m_d_C;

        // Compare the data.
        ASSERT_TRUE(compare(m_h_C.data(), m_h_C_ref.data(), m_h_C.size()));
    }

    // void
    // MeasurePerformance(cudaError_t (*launch_gemm)(T const*, T*, unsigned int,
    //                                               unsigned int,
    //                                               cudaStream_t),
    //                    unsigned int num_repeats = 20,
    //                    unsigned int num_warmups = 20)
    // {
    //     // GTEST_COUT << "M: " << m_M << " N: " << m_N << std::endl;

    //     // // Query deive name and peak memory bandwidth.
    //     // int device_id{0};
    //     // CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    //     // cudaDeviceProp device_prop;
    //     // CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop,
    //     device_id));
    //     // GTEST_COUT << "Device Name: " << device_prop.name << std::endl;
    //     // float const
    //     // memory_size{static_cast<float>(device_prop.totalGlobalMem) /
    //     //                         (1 << 30)};
    //     // GTEST_COUT << "Memory Size: " << memory_size << " GB" <<
    //     std::endl;
    //     // float const peak_bandwidth{
    //     //     static_cast<float>(2.0f * device_prop.memoryClockRate *
    //     //                        (device_prop.memoryBusWidth / 8) / 1.0e6)};
    //     // GTEST_COUT << "Peak Bandwitdh: " << peak_bandwidth << " GB/s"
    //     //            << std::endl;

    //     // auto const function{std::bind(launch_gemm,
    //     // thrust::raw_pointer_cast(m_d_src.data()),
    //     // thrust::raw_pointer_cast(m_d_dst.data()),
    //     //                               m_M, m_N, std::placeholders::_1)};
    //     // float const latency{measure_performance<T>(function, m_stream)};
    //     // GTEST_COUT << "Latency: " << latency << " ms" << std::endl;
    //     // GTEST_COUT << "Effective Bandwidth: "
    //     //            << convert_latency_to_effective_bandwidth<T>(latency,
    //     m_M,
    //     //                                                         m_N)
    //     //            << " GB/s" << std::endl;
    //     // GTEST_COUT << "Peak Bandwidth Percentage: "
    //     //            << 100.0f *
    //     // convert_latency_to_effective_bandwidth<T>(latency,
    //     //                                                             m_M,
    //     m_N)
    //     //                                                             /
    //     //                   peak_bandwidth
    //     //            << "%" << std::endl;
    // }

    char m_transa;
    char m_transb;

    int m_m;
    int m_n;
    int m_k;

    int m_lda;
    int m_ldb;
    int m_ldc;

    Alpha m_alpha;
    Beta m_beta;

    cudaStream_t m_stream;

    thrust::host_vector<TA> m_h_A;
    thrust::host_vector<TB> m_h_B;
    thrust::host_vector<TC> m_h_C;
    thrust::host_vector<TC> m_h_C_ref;

    thrust::device_vector<TA> m_d_A;
    thrust::device_vector<TB> m_d_B;
    thrust::device_vector<TC> m_d_C;

    // Random seed.
    unsigned int m_seed;
    std::default_random_engine m_random_engine;
};

class TestGeneralMatrixMultiplicationFloat
    : public TestGeneralMatrixMultiplication<float, float, float, float, float>
{
};

class TestGeneralMatrixMultiplicationDouble
    : public TestGeneralMatrixMultiplication<double, double, double, double,
                                             double>
{
};

class TestGeneralMatrixMultiplicationHalf
    : public TestGeneralMatrixMultiplication<cutlass::half_t, cutlass::half_t,
                                             cutlass::half_t, float, float>
{
};