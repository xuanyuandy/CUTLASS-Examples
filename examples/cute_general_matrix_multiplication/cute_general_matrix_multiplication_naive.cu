#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include "cute_general_matrix_multiplication.hpp"

// Modified from the official CuTe example:
// https://github.com/NVIDIA/cutlass/blob/e1cd8c7866dd6de02b66a89879795e7d7301aacc/examples/cute/tutorial/sgemm_1.cu#L52
// This implementation uses shared memory.
// This implementation can have shared memory bank conflicts when copying data
// from global memory to shared memory. This implementation does not register to
// cache the data from shared memory for local mma. This implementation does not
// explicitly use vectorized memory access. This implementation does not use
// TensorCore for mma.
template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class AThreadLayout, class TB, class BStride,
          class BSmemLayout, class BThreadLayout, class TC, class CStride,
          class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
static __global__ void general_matrix_multiplication_naive(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const* A, AStride stride_A,
    ASmemLayout smem_layout_A, AThreadLayout thread_layout_A, TB const* B,
    BStride stride_B, BSmemLayout smem_layout_B, BThreadLayout thread_layout_B,
    TC* C, CStride stride_C, CSmemLayout, CThreadLayout thread_layout_C,
    Alpha alpha, Beta beta)
{
    CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) ==
                         cute::Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    // Thread layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<AThreadLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BThreadLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<CThreadLayout>{});

    // Different thread layouts have to have the same number of threads.
    CUTE_STATIC_ASSERT_V(cute::size(thread_layout_A) ==
                         cute::size(thread_layout_B));
    CUTE_STATIC_ASSERT_V(cute::size(thread_layout_A) ==
                         cute::size(thread_layout_C));

    // CTA tiler has to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<CtaTiler>{});

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

    // Shared memory layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<ASmemLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BSmemLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<CSmemLayout>{});

    // Shared memory layouts have to match CTA tiler.
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_A) ==
                         cute::size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_A) ==
                         cute::size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(cute::size<0>(smem_layout_B) ==
                         cute::size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(cute::size<1>(smem_layout_B) ==
                         cute::size<2>(cta_tiler)); // BLK_K

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

    // Full tensor.
    // There are four scenarios for the full tensor.
    // 1. The shape of A is (M, K) and the shape of B is (K, N).
    //    Then A is (M, K) column-major and B is (K, N) column-major.
    //    Then A is (M, K) column-major and B is (N, K) row-major.
    // 2. The shape of transposed A is (M, K) and the shape of B is (K, N).
    //    Then A is (K, M) column-major and B is (K, N) column-major.
    //    Then A is (M, K) row-major and B is (N, K) row-major.
    // 3. The shape of A is (M, K) and the shape of transposed B is (K, N).
    //    Then A is (M, K) column-major and B is (N, K) column-major.
    // 4. The shape of transposed A is (M, K) and the shape of transposed B is
    // (K, N).
    //    Then A is (K, M) column-major and B is (N, K) column-major.
    //    Then A is (M, K) row-major and B is (N, K) column-major.
    auto global_full_tensor_A{cute::make_tensor(cute::make_gmem_ptr(A),
                                                cute::select<0, 2>(shape_MNK),
                                                stride_A)}; // (M, K)
    auto global_full_tensor_B{cute::make_tensor(cute::make_gmem_ptr(B),
                                                cute::select<1, 2>(shape_MNK),
                                                stride_B)}; // (N, K)
    // C is always (M, N) column-major.
    auto global_full_tensor_C{cute::make_tensor(cute::make_gmem_ptr(C),
                                                cute::select<0, 1>(shape_MNK),
                                                stride_C)}; // (M, N)

    // CTA index.
    // We used 3D index instead of 2D index because, as we will see later,
    // it will be convenient for the block selection, especially for the input
    // tensors A and B.
    auto cta_coord{
        cute::make_coord(blockIdx.x, blockIdx.y, cute::_)}; // (m, n, :)

    // Block selection.
    // With Step<_1, X, _1>{}, the second mode in the cta_tiler is ignored,
    // thus the tiler becomes (BLK_M, BLK_K).
    // In addition, because the the second mode is ignored, the cta_coord
    // becomes (m, :). So we will not select in the second mode.
    // The resulting local_tile is (BLK_M, BLK_K, k) where k is the number of
    // tiles to repeat and BLK_K * k = K if K is divisible by BLK_K.
    auto global_block_tensor_A{
        cute::local_tile(global_full_tensor_A, cta_tiler, cta_coord,
                         cute::Step<cute::Int<1>, cute::X,
                                    cute::Int<1>>{})}; // (BLK_M, BLK_K, k)
    // With Step<X, _1, _1>{}, the first mode in the cta_tiler is ignored,
    // thus the tiler becomes (BLK_N, BLK_K).
    // In addition, because the the first mode is ignored, the cta_coord
    // becomes (n, :). So we will not select in the first mode.
    // The resulting local_tile is (BLK_N, BLK_K, k) where k is the number of
    // tiles to repeat and BLK_K * k = K if K is divisible by BLK_K.
    auto global_block_tensor_B{
        cute::local_tile(global_full_tensor_B, cta_tiler, cta_coord,
                         cute::Step<cute::X, cute::Int<1>,
                                    cute::Int<1>>{})}; // (BLK_N, BLK_K, k)
    // With Step<_1, _1, X>{}, the third mode in the cta_tiler is ignored,
    // thus the tiler becomes (BLK_M, BLK_N).
    // In addition, because the the third mode is ignored, the cta_coord
    // becomes (m, n). So we will not select in the third mode.
    // The resulting local_tile is (BLK_M, BLK_N).
    auto global_block_tensor_C{cute::local_tile(
        global_full_tensor_C, cta_tiler, cta_coord,
        cute::Step<cute::Int<1>, cute::Int<1>, cute::X>{})}; // (BLK_M, BLK_N)

    // Shared memory buffers.
    __shared__ TA smem_A[cute::cosize_v<ASmemLayout>];
    __shared__ TB smem_B[cute::cosize_v<BSmemLayout>];
    // smem_layout_A and smem_layout_B are always column-major.
    // TODO: Add CUTE_STATIC_ASSERT to ensure the above conditions.
    auto smem_tensor_A{cute::make_tensor(cute::make_smem_ptr(smem_A),
                                         smem_layout_A)}; // (BLK_M, BLK_K)
    auto smem_tensor_B{cute::make_tensor(cute::make_smem_ptr(smem_B),
                                         smem_layout_B)}; // (BLK_N, BLK_K)

    // Partition the global_block_tensor_A and global_block_tensor_B across the
    // threads using the thread layout thread_layout_A and thread_layout_B.
    // Partition the smem_tensor_A and smem_tensor_B across the threads. This
    // will be used for copying the data from global memory to shared memory for
    // data reuse. Inner partition. This can ensure the memory access is
    // coalesced.
    auto thread_layout_A_global_block_tensor_A{cute::local_partition(
        global_block_tensor_A, thread_layout_A,
        threadIdx.x)}; // (BLK_M / THR_M, BLK_K / THR_K, k)
    auto thread_layout_B_global_block_tensor_B{cute::local_partition(
        global_block_tensor_B, thread_layout_B,
        threadIdx.x)}; // (BLK_N / THR_N, BLK_K / THR_K, k)
    auto thread_layout_A_smem_tensor_A{
        cute::local_partition(smem_tensor_A, thread_layout_A,
                              threadIdx.x)}; // (BLK_M / THR_M, BLK_K / THR_K)
    auto thread_layout_B_smem_tensor_B{
        cute::local_partition(smem_tensor_B, thread_layout_B,
                              threadIdx.x)}; // (BLK_N / THR_N, BLK_K / THR_K)

    CUTE_STATIC_ASSERT_V(
        cute::size<0>(thread_layout_A_global_block_tensor_A) ==
        cute::size<0>(thread_layout_A_smem_tensor_A)); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_A_global_block_tensor_A) ==
        cute::size<1>(thread_layout_A_smem_tensor_A)); // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(thread_layout_B_global_block_tensor_B) ==
        cute::size<0>(thread_layout_B_smem_tensor_B)); // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_B_global_block_tensor_B) ==
        cute::size<1>(thread_layout_B_smem_tensor_B)); // BLK_K / THR_K

    // Partition the smem_tensor_A and smem_tensor_B across the threads using
    // the thread layout thread_layout_C. Partition the global_block_tensor_C
    // across the threads. This will be used for the gemm computation. Inner
    // partition. Partition the smem_tensor_A (BLK_M, BLK_K) by the rows of
    // thread_layout_C. Different threads in the same column of thread_layout_C
    // will read the same data from smem_tensor_A. With Step<_1, X>{}, the
    // second mode in the thread_layout_C layout is ignored.
    // The threads in the same warp will read contiguous data from smem_tensor_A
    // resulting in free of shared memory bank conflict.
    auto thread_layout_C_smem_tensor_A{cute::local_partition(
        smem_tensor_A, thread_layout_C, threadIdx.x,
        cute::Step<cute::Int<1>, cute::X>{})}; // (BLK_M / THR_M,
                                               // BLK_K)
    // Partition the smem_tensor_B (BLK_N, BLK_K) by the cols of
    // thread_layout_C. Different threads in the same row of thread_layout_C
    // will read the same data from smem_tensor_B. With Step<X, _1>{}, the first
    // mode in the thread_layout_C layout is ignored.
    // The threads in the same warp will read the same data from the same
    // location on smem_tensor_B resulting in a broadcast and no efficiency
    // loss.
    auto thread_layout_C_smem_tensor_B{cute::local_partition(
        smem_tensor_B, thread_layout_C, threadIdx.x,
        cute::Step<cute::X, cute::Int<1>>{})}; // (BLK_N / THR_N,
                                               // BLK_K)
    // Partition the global_block_tensor_C (BLK_M, BLK_N) by the tile of
    // thread_layout_C.
    auto thread_layout_C_global_block_tensor_C{cute::local_partition(
        global_block_tensor_C, thread_layout_C, threadIdx.x,
        cute::Step<cute::Int<1>, cute::Int<1>>{})}; // (BLK_M / THR_M, BLK_N /
                                                    // THR_N)
    // This is the same as the above.
    // auto thread_layout_C_global_block_tensor_C{
    //     cute::local_partition(global_block_tensor_C, thread_layout_C,
    //                           threadIdx.x)}; // (BLK_M / THR_M, BLK_N /
    //                           THR_N)

    // Allocate the accumulators.
    // The layout is automatically compacted to the smallest possible layout to
    // avoid unnecessary memory/register usage.
    auto thread_layout_C_register_tensor_C{cute::make_tensor_like(
        thread_layout_C_global_block_tensor_C)}; // (BLK_M / THR_M, BLK_N /
                                                 // THR_N)

    CUTE_STATIC_ASSERT_V(
        cute::size<0>(thread_layout_C_smem_tensor_A) ==
        cute::size<0>(thread_layout_C_register_tensor_C)); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(thread_layout_C_smem_tensor_B) ==
        cute::size<1>(thread_layout_C_register_tensor_C)); // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(thread_layout_C_global_block_tensor_C) ==
        cute::size<0>(thread_layout_C_register_tensor_C)); // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_global_block_tensor_C) ==
        cute::size<1>(thread_layout_C_register_tensor_C)); // BLK_N / THR_N

    // Clear the accumulators.
    cute::clear(thread_layout_C_register_tensor_C);

#ifdef NO_BOUNDS_CHECK

#else
    // Create predicate tensors.
    auto thread_layout_A_predicate_tensor_A{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_layout_A_global_block_tensor_A),
                         cute::size<1>(thread_layout_A_global_block_tensor_A)),
        cute::make_stride(cute::Int<1>{}, cute::Int<0>{}))};
    auto thread_layout_B_predicate_tensor_B{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_layout_B_global_block_tensor_B),
                         cute::size<1>(thread_layout_B_global_block_tensor_B)),
        cute::make_stride(cute::Int<1>{}, cute::Int<0>{}))};
    auto thread_layout_C_predicate_tensor_C{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(thread_layout_C_global_block_tensor_C),
                         cute::size<1>(thread_layout_C_global_block_tensor_C)),
        cute::make_stride(
            cute::Int<1>{},
            cute::size<0>(thread_layout_C_global_block_tensor_C)))};
    // Create identity tensors.
    auto identity_tensor_A{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(smem_tensor_A), cute::size<1>(smem_tensor_A)))};
    auto identity_tensor_B{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(smem_tensor_B), cute::size<1>(smem_tensor_B)))};
    auto identity_tensor_C{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_block_tensor_C),
                         cute::size<1>(global_block_tensor_C)))};
    auto thread_layout_A_identity_tensor_A{
        cute::local_partition(identity_tensor_A, thread_layout_A,
                              threadIdx.x)}; // (BLK_M / THR_M, BLK_K / THR_K)
    auto thread_layout_B_identity_tensor_B{
        cute::local_partition(identity_tensor_B, thread_layout_B,
                              threadIdx.x)}; // (BLK_N / THR_N, BLK_K / THR_K)
    auto thread_layout_C_identity_tensor_C{
        cute::local_partition(identity_tensor_C, thread_layout_C,
                              threadIdx.x)}; // (BLK_M / THR_M, BLK_N / THR_N)

    CUTE_UNROLL
    for (auto m{0}; m < cute::size<0>(thread_layout_A_predicate_tensor_A); ++m)
    {
        thread_layout_A_predicate_tensor_A(m, 0) =
            cute::get<0>(thread_layout_A_identity_tensor_A(m, 0)) +
                blockIdx.x * cute::size<0>(smem_tensor_A) <
            cute::size<0>(shape_MNK);
    }
    CUTE_UNROLL
    for (auto n{0}; n < cute::size<0>(thread_layout_B_predicate_tensor_B); ++n)
    {
        thread_layout_B_predicate_tensor_B(n, 0) =
            cute::get<0>(thread_layout_B_identity_tensor_B(n, 0)) +
                blockIdx.y * cute::size<0>(smem_tensor_B) <
            cute::size<1>(shape_MNK);
    }
    CUTE_UNROLL
    for (auto m{0}; m < cute::size<0>(thread_layout_C_predicate_tensor_C); ++m)
    {
        CUTE_UNROLL
        for (auto n{0}; n < cute::size<1>(thread_layout_C_predicate_tensor_C);
             ++n)
        {
            thread_layout_C_predicate_tensor_C(m, n) =
                cute::get<0>(thread_layout_C_identity_tensor_C(m, n)) +
                        blockIdx.x * cute::size<0>(global_block_tensor_C) <
                    cute::size<0>(shape_MNK) &&
                cute::get<1>(thread_layout_C_identity_tensor_C(m, n)) +
                        blockIdx.y * cute::size<1>(global_block_tensor_C) <
                    cute::size<1>(shape_MNK);
        }
    }
#endif

    // Perform the gemm computation loop.
    auto const num_tiles_k{cute::size<2>(global_block_tensor_A)}; // k

    for (auto tile_idx_k{0}; tile_idx_k < num_tiles_k; ++tile_idx_k)
    {
#ifdef NO_BOUNDS_CHECK
        cute::copy(
            thread_layout_A_global_block_tensor_A(cute::_, cute::_, tile_idx_k),
            thread_layout_A_smem_tensor_A);
        cute::copy(
            thread_layout_B_global_block_tensor_B(cute::_, cute::_, tile_idx_k),
            thread_layout_B_smem_tensor_B);
#else
        // Clear the shared memory buffers.
        // This is necessary when predicates are used for copying data from
        // global memory to shared memory so that mma will not be affected by
        // the previous data in the unwanted region.
        cute::clear(thread_layout_A_smem_tensor_A);
        cute::clear(thread_layout_B_smem_tensor_B);

        CUTE_UNROLL
        for (auto k{0};
             k < cute::size<1>(thread_layout_A_global_block_tensor_A); ++k)
        {
            // Check the K dimension.
            if (cute::get<1>(thread_layout_A_identity_tensor_A(0, k)) +
                    tile_idx_k * cute::size<1>(smem_tensor_A) <
                cute::size<2>(shape_MNK))
            {
                cute::copy_if(thread_layout_A_predicate_tensor_A,
                              thread_layout_A_global_block_tensor_A(cute::_, k,
                                                                    tile_idx_k),
                              thread_layout_A_smem_tensor_A(cute::_, k));
            }
        }
        CUTE_UNROLL
        for (auto k{0};
             k < cute::size<1>(thread_layout_B_global_block_tensor_B); ++k)
        {
            // Check the K dimension.
            if (cute::get<1>(thread_layout_B_identity_tensor_B(0, k)) +
                    tile_idx_k * cute::size<1>(smem_tensor_B) <
                cute::size<2>(shape_MNK))
            {
                cute::copy_if(thread_layout_B_predicate_tensor_B,
                              thread_layout_B_global_block_tensor_B(cute::_, k,
                                                                    tile_idx_k),
                              thread_layout_B_smem_tensor_B(cute::_, k));
            }
        }
#endif
        // // Copy the data from global memory to shared memory for data reuse.
        // // This is the only place where shared memory bank conflicts can
        // happen,
        // // depending on the tensor layouts on the global memory.
        // // Copy the data from global_block_tensor_A to smem_tensor_A.
        // cute::copy(
        //     thread_layout_A_global_block_tensor_A(cute::_, cute::_,
        //     tile_idx_k), thread_layout_A_smem_tensor_A); // (BLK_M / THR_M,
        //     BLK_K / THR_K) ->
        //                                     // (BLK_M / THR_M, BLK_K / THR_K)
        // // Copy the data from global_block_tensor_B to smem_tensor_B.
        // cute::copy(
        //     thread_layout_B_global_block_tensor_B(cute::_, cute::_,
        //     tile_idx_k), thread_layout_B_smem_tensor_B); // (BLK_N / THR_N,
        //     BLK_K / THR_K) ->
        //                                     // (BLK_N / THR_N, BLK_K / THR_K)

        // Synchronize the threads to ensure the data copy is completed.
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();

        // Compute gemm on thread_layout_C thread-partitioned smem.
        // This implicitly uses the UniversalFMA GEMM atom.
        cute::gemm(thread_layout_C_smem_tensor_A, thread_layout_C_smem_tensor_B,
                   thread_layout_C_register_tensor_C); // (BLK_M / THR_M, BLK_N
                                                       // / THR_N) += (BLK_M /
                                                       // THR_M, BLK_K) * (BLK_N
                                                       // / THR_N, BLK_K)
        // This is equivalent to the above.
        // auto mma_atom{cute::MMA_Atom<cute::UniversalFMA<TC, TA, TB, TC>>{}};
        // cute::gemm(mma_atom, thread_layout_C_smem_tensor_A,
        //            thread_layout_C_smem_tensor_B,
        //            thread_layout_C_register_tensor_C); // (BLK_M / THR_M,
        //            BLK_N
        //                                                // / THR_N) += (BLK_M
        //                                                /
        //                                                // THR_M, BLK_K) *
        //                                                (BLK_N
        //                                                // / THR_N, BLK_K)

        __syncthreads();
    }

    // Scale and accumulate the result from the register tensor to the global
    // block tensor.
    // cute::axpby(
    //     alpha, thread_layout_C_register_tensor_C, beta,
    //     thread_layout_C_global_block_tensor_C); // (BLK_M / THR_M, BLK_N /
    //                                             // THR_N) = alpha * (BLK_M /
    //                                             // THR_M, BLK_N / THR_N) +
    //                                             beta
    //                                             // * (BLK_M / THR_M, BLK_N /
    //                                             // THR_N)
#ifdef NO_BOUNDS_CHECK
    cute::axpby(alpha, thread_layout_C_register_tensor_C, beta,
                thread_layout_C_global_block_tensor_C);
#else
    cute::axpby(alpha, thread_layout_C_register_tensor_C, beta,
                thread_layout_C_global_block_tensor_C,
                thread_layout_C_predicate_tensor_C);
#endif
}

template <class TA, class TB, class TC, class Alpha, class Beta, class AStride,
          class BStride, class CStride>
static cudaError_t gemm_base(int m, int n, int k, Alpha alpha, TA const* A,
                             int ldA, TB const* B, int ldB, Beta beta, TC* C,
                             int ldC, AStride stride_A, BStride stride_B,
                             CStride stride_C, cudaStream_t stream)
{
    // Define GEMM shape.
    auto const M{m};
    auto const N{n};
    auto const K{k};
    auto const gemm_shape{cute::make_shape(M, N, K)}; // (M, N, K)

    // Define CTA size.
    auto const bM{cute::Int<128 * 2 / sizeof(TA)>{}};
    auto const bN{cute::Int<128 * 2 / sizeof(TB)>{}};
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
        cute::make_shape(cute::Int<16>{}, cute::Int<8>{})}; // (THR_M, THR_K)
    auto const thread_shape_B{
        cute::make_shape(cute::Int<16>{}, cute::Int<8>{})}; // (THR_N, THR_K)
    auto const thread_shape_C{
        cute::make_shape(cute::Int<16>{}, cute::Int<8>{})}; // (THR_M, THR_N)
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

    // Launch the kernel.
    dim3 const block_dims{
        static_cast<unsigned int>(cute::size(thread_layout_C))};
    dim3 const grid_dims{
        static_cast<unsigned int>(cute::size(cute::ceil_div(M, bM))),
        static_cast<unsigned int>(cute::size(cute::ceil_div(N, bN)))};
    general_matrix_multiplication_naive<<<grid_dims, block_dims, 0, stream>>>(
        gemm_shape, cta_tiler, A, stride_A, smem_layout_A, thread_layout_A, B,
        stride_B, smem_layout_B, thread_layout_B, C, stride_C, smem_layout_C,
        thread_layout_C, alpha, beta);

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

    return gemm_base(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A,
                     stride_B, stride_C, stream);
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

    return gemm_base(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A,
                     stride_B, stride_C, stream);
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

    return gemm_base(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A,
                     stride_B, stride_C, stream);
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

    return gemm_base(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stride_A,
                     stride_B, stride_C, stream);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_gemm_naive(char transA, char transB, int m, int n, int k,
                              Alpha alpha, TA const* A, int ldA, TB const* B,
                              int ldB, Beta beta, TC* C, int ldC,
                              cudaStream_t stream)
{
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
template cudaError_t launch_gemm_naive<float, float, float, float, float>(
    char transA, char transB, int m, int n, int k, float alpha, float const* A,
    int ldA, float const* B, int ldB, float beta, float* C, int ldC,
    cudaStream_t stream);
template cudaError_t launch_gemm_naive<double, double, double, double, double>(
    char transA, char transB, int m, int n, int k, double alpha,
    double const* A, int ldA, double const* B, int ldB, double beta, double* C,
    int ldC, cudaStream_t stream);
template cudaError_t
launch_gemm_naive<cute::half_t, cute::half_t, cute::half_t, float, float>(
    char transA, char transB, int m, int n, int k, float alpha,
    cute::half_t const* A, int ldA, cute::half_t const* B, int ldB, float beta,
    cute::half_t* C, int ldC, cudaStream_t stream);
template cudaError_t launch_gemm_naive<cute::half_t, cute::half_t, cute::half_t,
                                       cute::half_t, cute::half_t>(
    char transA, char transB, int m, int n, int k, cute::half_t alpha,
    cute::half_t const* A, int ldA, cute::half_t const* B, int ldB,
    cute::half_t beta, cute::half_t* C, int ldC, cudaStream_t stream);