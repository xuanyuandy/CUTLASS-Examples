#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include "cute_general_matrix_multiplication.cuh"
#include "cute_general_matrix_multiplication.hpp"

constexpr int constexpr_log2(int n)
{
    return ((n < 2) ? 0 : 1 + constexpr_log2(n / 2));
}

template <class TA, class TB, class TC, class Alpha, class Beta, class AStride,
          class BStride, class CStride, class VectorTypeA, class VectorTypeB>
static cudaError_t gemm_base_gmem_tiled_copy_tiled_mma(
    int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B,
    int ldB, Beta beta, TC* C, int ldC, AStride stride_A, BStride stride_B,
    CStride stride_C, cudaStream_t stream)
{
    // Define GEMM shape.
    auto const M{m};
    auto const N{n};
    auto const K{k};
    auto const gemm_shape{cute::make_shape(M, N, K)}; // (M, N, K)

    // Define CTA size.
    auto const bM{cute::Int<128 * 4 / sizeof(TA)>{}};
    auto const bN{cute::Int<128 * 4 / sizeof(TB)>{}};
    auto const bK{cute::Int<32>{}};
    auto const cta_tiler{cute::make_shape(bM, bN, bK)}; // (BLK_M, BLK_N, BLK_K)

    // Define smem layouts.
    // smem_layout_A is (BLK_M, BLK_K) column-major.
    // smem_layout_B is (BLK_N, BLK_K) column-major.
    // smem_layout_C is (BLK_M, BLK_N) column-major.
    auto const smem_shape_A{cute::make_shape(bM, bK)}; // (BLK_M, BLK_K)
    auto const smem_stride_A{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(smem_shape_A))}; // column-major
    auto const smem_layout_A{
        cute::make_layout(smem_shape_A, smem_stride_A)}; // (BLK_M, BLK_K)
    auto const smem_shape_B{cute::make_shape(bN, bK)};   // (BLK_N, BLK_K)
    auto const smem_stride_B{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(smem_shape_B))}; // column-major
    auto const smem_layout_B{
        cute::make_layout(smem_shape_B, smem_stride_B)}; // (BLK_N, BLK_K)
    auto const smem_shape_C{cute::make_shape(bM, bN)};   // (BLK_M, BLK_N)
    auto const smem_stride_C{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(smem_shape_C))}; // column-major
    auto const smem_layout_C{
        cute::make_layout(smem_shape_C, smem_stride_C)}; // (BLK_M, BLK_N)

    // Define thread layouts.
    auto const thread_shape_A{
        cute::make_shape(cute::Int<32>{}, cute::Int<8>{})}; // (THR_M, THR_K)
    auto const thread_shape_B{
        cute::make_shape(cute::Int<32>{}, cute::Int<8>{})}; // (THR_N, THR_K)
    auto const thread_shape_C{
        cute::make_shape(cute::Int<16>{}, cute::Int<16>{})}; // (THR_M, THR_N)
    auto const thread_stride_A{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(thread_shape_A))}; // column-major
    auto const thread_stride_B{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(thread_shape_B))}; // column-major
    auto const thread_stride_C{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(thread_shape_C))}; // column-major
    auto const thread_layout_A{
        cute::make_layout(thread_shape_A, thread_stride_A)}; // (THR_M, THR_K)
    auto const thread_layout_B{
        cute::make_layout(thread_shape_B, thread_stride_B)}; // (THR_N, THR_K)
    auto const thread_layout_C{
        cute::make_layout(thread_shape_C, thread_stride_C)}; // (THR_M, THR_N)
    CUTE_STATIC_ASSERT_V(cute::size(thread_layout_A) ==
                         cute::size(thread_layout_B));
    CUTE_STATIC_ASSERT_V(cute::size(thread_layout_A) ==
                         cute::size(thread_layout_C));

    // CTA tiler has to be divisible by the thread layouts.
    CUTE_STATIC_ASSERT_V(cute::size<0>(cta_tiler) %
                             cute::size<0>(thread_layout_A) ==
                         cute::Int<0>{}); // BLK_M % THR_M == 0
    CUTE_STATIC_ASSERT_V(cute::size<2>(cta_tiler) %
                             cute::size<1>(thread_layout_A) ==
                         cute::Int<0>{}); // BLK_K % THR_K == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) %
                             cute::size<0>(thread_layout_B) ==
                         cute::Int<0>{}); // BLK_N % THR_N == 0
    CUTE_STATIC_ASSERT_V(cute::size<2>(cta_tiler) %
                             cute::size<1>(thread_layout_B) ==
                         cute::Int<0>{}); // BLK_K % THR_K == 0
    CUTE_STATIC_ASSERT_V(cute::size<0>(cta_tiler) %
                             cute::size<0>(thread_layout_C) ==
                         cute::Int<0>{}); // BLK_M % THR_M == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) %
                             cute::size<1>(thread_layout_C) ==
                         cute::Int<0>{}); // BLK_N % THR_N == 0

    // Shared memory layouts have to be divisible by the thread layouts.
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_A) %
                             cute::size<0>(thread_layout_A) ==
                         cute::Int<0>{}); // BLK_M % THR_M == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_A) %
                             cute::size<1>(thread_layout_A) ==
                         cute::Int<0>{}); // BLK_K % THR_K == 0
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_B) %
                             cute::size<0>(thread_layout_B) ==
                         cute::Int<0>{}); // BLK_N % THR_N == 0
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_B) %
                             cute::size<1>(thread_layout_B) ==
                         cute::Int<0>{}); // BLK_K % THR_K == 0

    constexpr auto NUM_VECTOR_ELEMENTS_A{sizeof(VectorTypeA) / sizeof(TA)};
    auto const vector_shape_A{
        cute::make_shape(cute::Int<NUM_VECTOR_ELEMENTS_A>{},
                         cute::Int<1>{})}; // (NUM_VECTOR_ELEMENTS_A, 1)
    auto const vector_stride_A{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(vector_shape_A))}; // column-major
    auto const vector_layout_A{cute::make_layout(
        vector_shape_A, vector_stride_A)}; // (NUM_VECTOR_ELEMENTS_A, 1)
    auto copy_A{cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<VectorTypeA>, TA>{},
        thread_layout_A,
        vector_layout_A)}; // Thread layout: (THR_M, THR_K) Value layout:
                           // (NUM_VECTOR_ELEMENTS_A, 1)
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(smem_layout_A) %
            (cute::size<0>(thread_layout_A) * cute::size<0>(vector_layout_A)) ==
        cute::Int<0>{}); // BLK_M % (THR_M * NUM_VECTOR_ELEMENTS_A) == 0

    constexpr auto NUM_VECTOR_ELEMENTS_B{sizeof(VectorTypeB) / sizeof(TB)};
    auto const vector_shape_B{
        cute::make_shape(cute::Int<NUM_VECTOR_ELEMENTS_B>{},
                         cute::Int<1>{})}; // (NUM_VECTOR_ELEMENTS_B, 1)
    auto const vector_stride_B{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(vector_shape_B))}; // column-major
    auto const vector_layout_B{cute::make_layout(
        vector_shape_B, vector_stride_B)}; // (NUM_VECTOR_ELEMENTS_B, 1)
    auto copy_B{cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<VectorTypeB>, TB>{},
        thread_layout_B,
        vector_layout_B)}; // Thread layout: (THR_N, THR_K) Value layout:
                           // (NUM_VECTOR_ELEMENTS_B, 1)
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(smem_layout_B) %
            (cute::size<0>(thread_layout_B) * cute::size<0>(vector_layout_B)) ==
        cute::Int<0>{}); // BLK_N % (THR_N * NUM_VECTOR_ELEMENTS_B) == 0

    auto mma{cute::make_tiled_mma(cute::UniversalFMA<TA, TB, TC>{},
                                  thread_layout_C)};

    // Swizzle parameters.
    constexpr int NUM_BASE_BITS_A{constexpr_log2(NUM_VECTOR_ELEMENTS_A)};
    constexpr int NUM_MASK_BITS_A{constexpr_log2(32 * 4 / sizeof(TA)) -
                                  NUM_BASE_BITS_A};
    constexpr int NUM_SHIFT_BITS_A{constexpr_log2(bM) - NUM_BASE_BITS_A};

    constexpr int NUM_BASE_BITS_B{constexpr_log2(NUM_VECTOR_ELEMENTS_B)};
    constexpr int NUM_MASK_BITS_B{constexpr_log2(32 * 4 / sizeof(TB)) -
                                  NUM_BASE_BITS_B};
    constexpr int NUM_SHIFT_BITS_B{constexpr_log2(bN) - NUM_BASE_BITS_B};

    auto const swizzle_A{
        cute::Swizzle<NUM_MASK_BITS_A, NUM_BASE_BITS_A, NUM_SHIFT_BITS_A>{}};
    auto const swizzle_B{
        cute::Swizzle<NUM_MASK_BITS_B, NUM_BASE_BITS_B, NUM_SHIFT_BITS_B>{}};

    // In fact, for some layouts, swizzles are not needed if no transpose is
    // performed.
    // But it should not reduce the performance even if the transpose is not
    // performed.
    auto const smem_layout_A_swizzled{
        cute::composition(swizzle_A, smem_layout_A)};
    auto const smem_layout_B_swizzled{
        cute::composition(swizzle_B, smem_layout_B)};

    // Launch the kernel.
    dim3 const block_dims{
        static_cast<unsigned int>(cute::size(thread_layout_C))};
    dim3 const grid_dims{
        static_cast<unsigned int>(cute::size(cute::ceil_div(M, bM))),
        static_cast<unsigned int>(cute::size(cute::ceil_div(N, bN)))};
    general_matrix_multiplication_gmem_tiled_copy_tiled_mma_sm70_pipeline<<<
        grid_dims, block_dims, 0, stream>>>(
        gemm_shape, cta_tiler, A, stride_A, smem_layout_A_swizzled,
        thread_layout_A, copy_A, B, stride_B, smem_layout_B_swizzled,
        thread_layout_B, copy_B, C, stride_C, smem_layout_C, thread_layout_C,
        mma, alpha, beta);

    return cudaGetLastError();
}

// The shape of A is (M, K) and the shape of B is (K, N).
// Then A is (M, K) column-major and B is (K, N) column-major.
// Then A is (M, K) column-major and B is (N, K) row-major.
template <class TA, class TB, class TC, class Alpha, class Beta>
static cudaError_t gemm_nn(int m, int n, int k, Alpha alpha, TA const* A,
                           int ldA, TB const* B, int ldB, Beta beta, TC* C,
                           int ldC, cudaStream_t stream)
{
    // Define global memory layouts.
    // A is (M, K) column-major.
    auto const stride_A{cute::make_stride(cute::Int<1>{}, ldA)}; // column-major
    // B is (N, K) row-major.
    auto const stride_B{cute::make_stride(ldB, cute::Int<1>{})}; // row-major
    // C is (M, N) column-major.
    auto const stride_C{cute::make_stride(cute::Int<1>{}, ldC)}; // column-major

    using VectorTypeA = cute::uint128_t;
    using VectorTypeB = TB;

    return gemm_base_gmem_tiled_copy_tiled_mma<
        TA, TB, TC, Alpha, Beta, decltype(stride_A), decltype(stride_B),
        decltype(stride_C), VectorTypeA, VectorTypeB>(
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A, stride_B,
        stride_C, stream);
}

// The shape of A is (M, K) and the shape of transposed B is (K, N).
// Then A is (M, K) column-major and B is (N, K) column-major.
// The smem_A is (BLK_M, BLK_K) column-major and smem_B is (BLK_N, BLK_K)
// column-major.
template <class TA, class TB, class TC, class Alpha, class Beta>
static cudaError_t gemm_nt(int m, int n, int k, Alpha alpha, TA const* A,
                           int ldA, TB const* B, int ldB, Beta beta, TC* C,
                           int ldC, cudaStream_t stream)
{
    // Define global memory layouts.
    // A is (M, K) column-major.
    auto const stride_A{cute::make_stride(cute::Int<1>{}, ldA)}; // column-major
    // B is (N, K) column-major.
    auto const stride_B{cute::make_stride(cute::Int<1>{}, ldB)}; // column-major
    // C is (M, N) column-major.
    auto const stride_C{cute::make_stride(cute::Int<1>{}, ldC)}; // column-major

    using VectorTypeA = cute::uint128_t;
    using VectorTypeB = cute::uint128_t;

    return gemm_base_gmem_tiled_copy_tiled_mma<
        TA, TB, TC, Alpha, Beta, decltype(stride_A), decltype(stride_B),
        decltype(stride_C), VectorTypeA, VectorTypeB>(
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A, stride_B,
        stride_C, stream);
}

// The shape of transposed A is (M, K) and the shape of B is (K, N).
// Then A is (K, M) column-major and B is (K, N) column-major.
// Then A is (M, K) row-major and B is (N, K) row-major.
template <class TA, class TB, class TC, class Alpha, class Beta>
static cudaError_t gemm_tn(int m, int n, int k, Alpha alpha, TA const* A,
                           int ldA, TB const* B, int ldB, Beta beta, TC* C,
                           int ldC, cudaStream_t stream)
{
    // Define global memory layouts.
    // A is (M, K) row-major.
    auto const stride_A{cute::make_stride(ldA, cute::Int<1>{})}; // row-major
    // B is (N, K) row-major.
    auto const stride_B{cute::make_stride(ldB, cute::Int<1>{})}; // row-major
    // C is (M, N) column-major.
    auto const stride_C{cute::make_stride(cute::Int<1>{}, ldC)}; // column-major

    // Because the shared memory layout is (BLK_M, BLK_K) column-major and
    // the global memory layout is (M, K) row-major, a transpose is needed and
    // vectorized memory copy is not possible. This transpose will result in
    // shared memory bank conflicts if not padding or swizzling is used. Another
    // strategy is to make the shared memory layout (BLK_M, BLK_K) row-major and
    // then we could perform vectorized memory copy. However, even with
    // swizzling or padding, there are can still be shared memory bank
    // conflicts. See https://leimao.github.io/blog/CuTe-Swizzle/ for more
    // information. So it is matter of experimentation to find the best strategy
    // for a specific problem. For this example, we will use the first strategy
    // without thoroughly investigating which strategy is better.
    using VectorTypeA = TA;
    using VectorTypeB = TB;

    return gemm_base_gmem_tiled_copy_tiled_mma<
        TA, TB, TC, Alpha, Beta, decltype(stride_A), decltype(stride_B),
        decltype(stride_C), VectorTypeA, VectorTypeB>(
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A, stride_B,
        stride_C, stream);
}

// The shape of transposed A is (M, K) and the shape of transposed B is (K, N).
//    Then A is (K, M) column-major and B is (N, K) column-major.
//    Then A is (M, K) row-major and B is (N, K) column-major.
template <class TA, class TB, class TC, class Alpha, class Beta>
static cudaError_t gemm_tt(int m, int n, int k, Alpha alpha, TA const* A,
                           int ldA, TB const* B, int ldB, Beta beta, TC* C,
                           int ldC, cudaStream_t stream)
{
    // Define global memory layouts.
    // A is (M, K) row-major.
    auto const stride_A{cute::make_stride(ldA, cute::Int<1>{})}; // row-major
    // B is (N, K) column-major.
    auto const stride_B{cute::make_stride(cute::Int<1>{}, ldB)}; // column-major
    // C is (M, N) column-major.
    auto const stride_C{cute::make_stride(cute::Int<1>{}, ldC)}; // column-major

    using VectorTypeA = TA;
    using VectorTypeB = cute::uint128_t;

    return gemm_base_gmem_tiled_copy_tiled_mma<
        TA, TB, TC, Alpha, Beta, decltype(stride_A), decltype(stride_B),
        decltype(stride_C), VectorTypeA, VectorTypeB>(
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A, stride_B,
        stride_C, stream);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_gemm_naive_gmem_tiled_copy_tiled_mma_sm70_pipeline(
    char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A,
    int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC,
    cudaStream_t stream)
{
    // To ensure vectorized memory access, the values of m, n, and k are
    // constrained to be:
    if (m * sizeof(TA) % 128 != 0 || k * sizeof(TA) % 128 != 0)
    {
        return cudaErrorNotSupported;
    }
    if (k * sizeof(TB) % 128 != 0 || n * sizeof(TB) % 128 != 0)
    {
        return cudaErrorNotSupported;
    }
    if (m * sizeof(TC) % 128 != 0 || n * sizeof(TC) % 128 != 0)
    {
        return cudaErrorNotSupported;
    }
    // To ensure data alignment, the values of ldA, ldB, and ldC are constrained
    // to be:
    if (ldA * sizeof(TA) % 128 != 0 || ldB * sizeof(TB) % 128 != 0 ||
        ldC * sizeof(TC) % 128 != 0)
    {
        return cudaErrorNotSupported;
    }

    if (transA == 'N' && transB == 'T')
    {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else if (transA == 'N' && transB == 'N')
    {
        return gemm_nn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else if (transA == 'T' && transB == 'N')
    {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else if (transA == 'T' && transB == 'T')
    {
        return gemm_tt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    else
    {
        return cudaErrorNotSupported;
    }
}

// Explicit instantiation
template cudaError_t launch_gemm_naive_gmem_tiled_copy_tiled_mma_sm70_pipeline<
    float, float, float, float, float>(char transA, char transB, int m, int n,
                                       int k, float alpha, float const* A,
                                       int ldA, float const* B, int ldB,
                                       float beta, float* C, int ldC,
                                       cudaStream_t stream);
template cudaError_t launch_gemm_naive_gmem_tiled_copy_tiled_mma_sm70_pipeline<
    double, double, double, double, double>(char transA, char transB, int m,
                                            int n, int k, double alpha,
                                            double const* A, int ldA,
                                            double const* B, int ldB,
                                            double beta, double* C, int ldC,
                                            cudaStream_t stream);
template cudaError_t launch_gemm_naive_gmem_tiled_copy_tiled_mma_sm70_pipeline<
    cute::half_t, cute::half_t, cute::half_t, float, float>(
    char transA, char transB, int m, int n, int k, float alpha,
    cute::half_t const* A, int ldA, cute::half_t const* B, int ldB, float beta,
    cute::half_t* C, int ldC, cudaStream_t stream);
template cudaError_t launch_gemm_naive_gmem_tiled_copy_tiled_mma_sm70_pipeline<
    cute::half_t, cute::half_t, cute::half_t, cute::half_t, cute::half_t>(
    char transA, char transB, int m, int n, int k, cute::half_t alpha,
    cute::half_t const* A, int ldA, cute::half_t const* B, int ldB,
    cute::half_t beta, cute::half_t* C, int ldC, cudaStream_t stream);