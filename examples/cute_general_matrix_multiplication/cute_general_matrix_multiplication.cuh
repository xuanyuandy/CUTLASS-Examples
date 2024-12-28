#ifndef CUTE_GENERAL_MATRIX_MULTIPLICATION_CUH
#define CUTE_GENERAL_MATRIX_MULTIPLICATION_CUH

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

// Use tiled copy and tiled MMA.
template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class AThreadLayout, class TiledCopyA, class TB,
          class BStride, class BSmemLayout, class BThreadLayout,
          class TiledCopyB, class TC, class CStride, class CSmemLayout,
          class CThreadLayout, class TiledMMA, class Alpha, class Beta>
__global__ void general_matrix_multiplication_tiled_copy_tiled_mma(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const* A, AStride stride_A,
    ASmemLayout smem_layout_A, AThreadLayout, TiledCopyA copy_A, TB const* B,
    BStride stride_B, BSmemLayout smem_layout_B, BThreadLayout,
    TiledCopyB copy_B, TC* C, CStride stride_C, CSmemLayout, CThreadLayout,
    TiledMMA mma, Alpha alpha, Beta beta)
{
    CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) ==
                         cute::Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    // Thread layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<AThreadLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BThreadLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<CThreadLayout>{});

    // CTA tiler has to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<CtaTiler>{});

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
    // sA and sB are always column-major.
    // TODO: Add CUTE_STATIC_ASSERT to ensure the above conditions.
    auto smem_tensor_A{cute::make_tensor(cute::make_smem_ptr(smem_A),
                                         smem_layout_A)}; // (BLK_M, BLK_K)
    auto smem_tensor_B{cute::make_tensor(cute::make_smem_ptr(smem_B),
                                         smem_layout_B)}; // (BLK_N, BLK_K)

    // Partition via tiled copy.
    auto thread_copy_A{copy_A.get_slice(threadIdx.x)};
    auto thread_layout_A_global_block_tensor_A{thread_copy_A.partition_S(
        global_block_tensor_A)}; // (CPY, CPY_M, CPY_K, k)
    auto thread_layout_A_smem_tensor_A{
        thread_copy_A.partition_D(smem_tensor_A)}; // (CPY, CPY_M, CPY_K)
    auto thread_copy_B{copy_B.get_slice(threadIdx.x)};
    auto thread_layout_B_global_block_tensor_B{thread_copy_B.partition_S(
        global_block_tensor_B)}; // (CPY, CPY_N, CPY_K, k)
    auto thread_layout_B_smem_tensor_B{
        thread_copy_B.partition_D(smem_tensor_B)}; // (CPY, CPY_N, CPY_K)

    // Partition via MMA.
    auto thread_mma{mma.get_slice(threadIdx.x)};
    auto thread_layout_C_smem_tensor_A{
        thread_mma.partition_A(smem_tensor_A)}; // (MMA, MMA_M, MMA_K)
    auto thread_layout_C_smem_tensor_B{
        thread_mma.partition_B(smem_tensor_B)}; // (MMA, MMA_N, MMA_K)
    auto thread_layout_C_global_block_tensor_C{
        thread_mma.partition_C(global_block_tensor_C)}; // (MMA, MMA_M, MMA_N)

    // Allocate the accumulators.
    // The layout is automatically compacted to the smallest possible layout to
    // avoid unnecessary memory/register usage.
    auto thread_layout_C_register_tensor_C{cute::make_fragment_like(
        thread_layout_C_global_block_tensor_C)}; // (MMA, MMA_M, MMA_N)

    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_smem_tensor_A) ==
        cute::size<1>(thread_layout_C_register_tensor_C)); // MMA_M
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_smem_tensor_B) ==
        cute::size<2>(thread_layout_C_register_tensor_C)); // MMA_N
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_global_block_tensor_C) ==
        cute::size<1>(thread_layout_C_register_tensor_C)); // MMA_M
    CUTE_STATIC_ASSERT_V(
        cute::size<2>(thread_layout_C_global_block_tensor_C) ==
        cute::size<2>(thread_layout_C_register_tensor_C)); // MMA_N

    // Clear the accumulators.
    cute::clear(thread_layout_C_register_tensor_C);

    // Create identity tensors.
    auto identity_tensor_A{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(smem_tensor_A), cute::size<1>(smem_tensor_A)))};
    auto identity_tensor_B{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(smem_tensor_B), cute::size<1>(smem_tensor_B)))};
    auto identity_tensor_C{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_block_tensor_C),
                         cute::size<1>(global_block_tensor_C)))};
    auto thread_layout_A_identity_tensor_A{
        thread_copy_A.partition_S(identity_tensor_A)}; // (CPY, CPY_M, CPY_K)
    auto thread_layout_B_identity_tensor_B{
        thread_copy_B.partition_S(identity_tensor_B)}; // (CPY, CPY_N, CPY_K)
    auto thread_layout_C_identity_tensor_C{
        thread_mma.partition_C(identity_tensor_C)}; // (MMA, MMA_M, MMA_N)
    // Create predicate tensors.
    auto thread_layout_A_predicate_tensor_A{cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(thread_layout_A_identity_tensor_A),
                         cute::size<2>(thread_layout_A_identity_tensor_A)))};
    auto thread_layout_B_predicate_tensor_B{cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(thread_layout_B_identity_tensor_B),
                         cute::size<2>(thread_layout_B_identity_tensor_B)))};
    auto thread_layout_C_predicate_tensor_C{cute::make_tensor<bool>(
        cute::shape(thread_layout_C_identity_tensor_C))};

    CUTE_UNROLL
    for (auto m{0}; m < cute::size<0>(thread_layout_A_predicate_tensor_A); ++m)
    {
        CUTE_UNROLL
        for (auto k{0}; k < cute::size<1>(thread_layout_A_predicate_tensor_A);
             ++k)
        {
            thread_layout_A_predicate_tensor_A(m, k) =
                cute::get<0>(thread_layout_A_identity_tensor_A(0, m, k)) +
                        blockIdx.x * cute::size<0>(smem_tensor_A) <
                    cute::size<0>(shape_MNK) &&
                cute::get<1>(thread_layout_A_identity_tensor_A(0, m, k)) +
                        blockIdx.y * cute::size<1>(smem_tensor_A) <
                    cute::size<2>(shape_MNK);
        }
    }
    CUTE_UNROLL
    for (auto n{0}; n < cute::size<0>(thread_layout_B_predicate_tensor_B); ++n)
    {
        CUTE_UNROLL
        for (auto k{0}; k < cute::size<1>(thread_layout_B_predicate_tensor_B);
             ++k)
        {
            thread_layout_B_predicate_tensor_B(n, k) =
                cute::get<0>(thread_layout_B_identity_tensor_B(0, n, k)) +
                        blockIdx.y * cute::size<0>(smem_tensor_B) <
                    cute::size<1>(shape_MNK) &&
                cute::get<1>(thread_layout_B_identity_tensor_B(0, n, k)) +
                        blockIdx.x * cute::size<1>(smem_tensor_B) <
                    cute::size<2>(shape_MNK);
        }
    }
    CUTE_UNROLL
    for (auto i{0}; i < cute::size(thread_layout_C_predicate_tensor_C); ++i)
    {
        thread_layout_C_predicate_tensor_C(i) =
            cute::get<0>(thread_layout_C_identity_tensor_C(i)) +
                    blockIdx.x * cute::size<0>(global_block_tensor_C) <
                cute::size<0>(shape_MNK) &&
            cute::get<1>(thread_layout_C_identity_tensor_C(i)) +
                    blockIdx.y * cute::size<1>(global_block_tensor_C) <
                cute::size<1>(shape_MNK);
    }

    // Perform the gemm computation loop.
    auto const num_tiles_k{cute::size<2>(global_block_tensor_A)}; // k

    for (auto tile_idx_k{0}; tile_idx_k < num_tiles_k; ++tile_idx_k)
    {
        // Clear the shared memory buffers.
        // This is necessary when predicates are used for copying data from
        // global memory to shared memory so that mma will not be affected by
        // the previous data in the unwanted region.
        cute::clear(thread_layout_A_smem_tensor_A);
        cute::clear(thread_layout_B_smem_tensor_B);

        cute::copy_if(copy_A, thread_layout_A_predicate_tensor_A,
                      thread_layout_A_global_block_tensor_A(
                          cute::_, cute::_, cute::_, tile_idx_k),
                      thread_layout_A_smem_tensor_A);
        cute::copy_if(copy_B, thread_layout_B_predicate_tensor_B,
                      thread_layout_B_global_block_tensor_B(
                          cute::_, cute::_, cute::_, tile_idx_k),
                      thread_layout_B_smem_tensor_B);

        // Synchronize the threads to ensure the data copy is completed.
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();

        // Compute gemm on thread_layout_C thread-partitioned smem.
        // This implicitly uses the UniversalFMA GEMM atom.
        cute::gemm(mma, thread_layout_C_smem_tensor_A,
                   thread_layout_C_smem_tensor_B,
                   thread_layout_C_register_tensor_C); // (BLK_M / THR_M, BLK_N
                                                       // / THR_N) += (BLK_M /
                                                       // THR_M, BLK_K) * (BLK_N
                                                       // / THR_N, BLK_K)

        __syncthreads();
    }

    // Scale and accumulate the result from the register tensor to the global
    // block tensor.
    // There does not seem to be a tiled axpby existing yet.
    cute::axpby(alpha, thread_layout_C_register_tensor_C, beta,
                thread_layout_C_global_block_tensor_C,
                thread_layout_C_predicate_tensor_C);
}

// Use tiled copy and tiled MMA.
template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class AThreadLayout, class TiledCopyA, class TB,
          class BStride, class BSmemLayout, class BThreadLayout,
          class TiledCopyB, class TC, class CStride, class CSmemLayout,
          class CThreadLayout, class TiledMMA, class Alpha, class Beta>
__global__ void
general_matrix_multiplication_tiled_copy_tiled_mma_sm70_pipeline(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const* A, AStride stride_A,
    ASmemLayout smem_layout_A, AThreadLayout, TiledCopyA copy_A, TB const* B,
    BStride stride_B, BSmemLayout smem_layout_B, BThreadLayout,
    TiledCopyB copy_B, TC* C, CStride stride_C, CSmemLayout, CThreadLayout,
    TiledMMA mma, Alpha alpha, Beta beta)
{
    CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) ==
                         cute::Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    // Thread layouts have to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<AThreadLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<BThreadLayout>{});
    CUTE_STATIC_ASSERT_V(cute::is_static<CThreadLayout>{});

    // CTA tiler has to be static.
    CUTE_STATIC_ASSERT_V(cute::is_static<CtaTiler>{});

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
    // sA and sB are always column-major.
    // TODO: Add CUTE_STATIC_ASSERT to ensure the above conditions.
    auto smem_tensor_A{cute::make_tensor(cute::make_smem_ptr(smem_A),
                                         smem_layout_A)}; // (BLK_M, BLK_K)
    auto smem_tensor_B{cute::make_tensor(cute::make_smem_ptr(smem_B),
                                         smem_layout_B)}; // (BLK_N, BLK_K)

    // Partition via tiled copy.
    auto thread_copy_A{copy_A.get_slice(threadIdx.x)};
    auto thread_layout_A_global_block_tensor_A{thread_copy_A.partition_S(
        global_block_tensor_A)}; // (CPY, CPY_M, CPY_K, k)
    auto thread_layout_A_smem_tensor_A{
        thread_copy_A.partition_D(smem_tensor_A)}; // (CPY, CPY_M, CPY_K)
    // Loading data from global memory to register while shared memory is being
    // loaded. Instead of using double shared memory buffer, we used more
    // registers for pipelining.
    auto thread_layout_A_register_tensor_A{cute::make_fragment_like(
        thread_layout_A_smem_tensor_A)}; // (CPY, CPY_M, CPY_K)
    auto thread_copy_B{copy_B.get_slice(threadIdx.x)};
    auto thread_layout_B_global_block_tensor_B{thread_copy_B.partition_S(
        global_block_tensor_B)}; // (CPY, CPY_N, CPY_K, k)
    auto thread_layout_B_smem_tensor_B{
        thread_copy_B.partition_D(smem_tensor_B)}; // (CPY, CPY_N, CPY_K)
    // Loading data from global memory to register while shared memory is being
    // loaded. Instead of using double shared memory buffer, we used more
    // registers for pipelining.
    auto thread_layout_B_register_tensor_B{cute::make_fragment_like(
        thread_layout_B_smem_tensor_B)}; // (CPY, CPY_N, CPY_K)

    // Partition via MMA.
    auto thread_mma{mma.get_slice(threadIdx.x)};
    auto thread_layout_C_smem_tensor_A{
        thread_mma.partition_A(smem_tensor_A)}; // (MMA, MMA_M, MMA_K)
    auto thread_layout_C_smem_tensor_B{
        thread_mma.partition_B(smem_tensor_B)}; // (MMA, MMA_N, MMA_K)
    auto thread_layout_C_global_block_tensor_C{
        thread_mma.partition_C(global_block_tensor_C)}; // (MMA, MMA_M, MMA_N)

    // Allocate the accumulators.
    // The layout is automatically compacted to the smallest possible layout to
    // avoid unnecessary memory/register usage.
    auto thread_layout_C_register_tensor_A{cute::make_fragment_like(
        thread_layout_C_smem_tensor_A)}; // (MMA, MMA_M, MMA_K)
    auto thread_layout_C_register_tensor_B{cute::make_fragment_like(
        thread_layout_C_smem_tensor_B)}; // (MMA, MMA_N, MMA_K)
    auto thread_layout_C_register_tensor_C{cute::make_fragment_like(
        thread_layout_C_global_block_tensor_C)}; // (MMA, MMA_M, MMA_N)

    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_smem_tensor_A) ==
        cute::size<1>(thread_layout_C_register_tensor_C)); // MMA_M
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_smem_tensor_B) ==
        cute::size<2>(thread_layout_C_register_tensor_C)); // MMA_N
    CUTE_STATIC_ASSERT_V(
        cute::size<1>(thread_layout_C_global_block_tensor_C) ==
        cute::size<1>(thread_layout_C_register_tensor_C)); // MMA_M
    CUTE_STATIC_ASSERT_V(
        cute::size<2>(thread_layout_C_global_block_tensor_C) ==
        cute::size<2>(thread_layout_C_register_tensor_C)); // MMA_N

    // Clear the accumulators.
    cute::clear(thread_layout_C_register_tensor_C);

    // Create identity tensors.
    auto identity_tensor_A{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(smem_tensor_A), cute::size<1>(smem_tensor_A)))};
    auto identity_tensor_B{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(smem_tensor_B), cute::size<1>(smem_tensor_B)))};
    auto identity_tensor_C{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_block_tensor_C),
                         cute::size<1>(global_block_tensor_C)))};
    auto thread_layout_A_identity_tensor_A{
        thread_copy_A.partition_S(identity_tensor_A)}; // (CPY, CPY_M, CPY_K)
    auto thread_layout_B_identity_tensor_B{
        thread_copy_B.partition_S(identity_tensor_B)}; // (CPY, CPY_N, CPY_K)
    auto thread_layout_C_identity_tensor_C{
        thread_mma.partition_C(identity_tensor_C)}; // (MMA, MMA_M, MMA_N)
    // Create predicate tensors.
    auto thread_layout_A_predicate_tensor_A{cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(thread_layout_A_identity_tensor_A),
                         cute::size<2>(thread_layout_A_identity_tensor_A)))};
    auto thread_layout_B_predicate_tensor_B{cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(thread_layout_B_identity_tensor_B),
                         cute::size<2>(thread_layout_B_identity_tensor_B)))};
    auto thread_layout_C_predicate_tensor_C{cute::make_tensor<bool>(
        cute::shape(thread_layout_C_identity_tensor_C))};

    CUTE_UNROLL
    for (auto m{0}; m < cute::size<0>(thread_layout_A_predicate_tensor_A); ++m)
    {
        CUTE_UNROLL
        for (auto k{0}; k < cute::size<1>(thread_layout_A_predicate_tensor_A);
             ++k)
        {
            thread_layout_A_predicate_tensor_A(m, k) =
                cute::get<0>(thread_layout_A_identity_tensor_A(0, m, k)) +
                        blockIdx.x * cute::size<0>(smem_tensor_A) <
                    cute::size<0>(shape_MNK) &&
                cute::get<1>(thread_layout_A_identity_tensor_A(0, m, k)) +
                        blockIdx.y * cute::size<1>(smem_tensor_A) <
                    cute::size<2>(shape_MNK);
        }
    }
    CUTE_UNROLL
    for (auto n{0}; n < cute::size<0>(thread_layout_B_predicate_tensor_B); ++n)
    {
        CUTE_UNROLL
        for (auto k{0}; k < cute::size<1>(thread_layout_B_predicate_tensor_B);
             ++k)
        {
            thread_layout_B_predicate_tensor_B(n, k) =
                cute::get<0>(thread_layout_B_identity_tensor_B(0, n, k)) +
                        blockIdx.y * cute::size<0>(smem_tensor_B) <
                    cute::size<1>(shape_MNK) &&
                cute::get<1>(thread_layout_B_identity_tensor_B(0, n, k)) +
                        blockIdx.x * cute::size<1>(smem_tensor_B) <
                    cute::size<2>(shape_MNK);
        }
    }
    CUTE_UNROLL
    for (auto i{0}; i < cute::size(thread_layout_C_predicate_tensor_C); ++i)
    {
        thread_layout_C_predicate_tensor_C(i) =
            cute::get<0>(thread_layout_C_identity_tensor_C(i)) +
                    blockIdx.x * cute::size<0>(global_block_tensor_C) <
                cute::size<0>(shape_MNK) &&
            cute::get<1>(thread_layout_C_identity_tensor_C(i)) +
                    blockIdx.y * cute::size<1>(global_block_tensor_C) <
                cute::size<1>(shape_MNK);
    }

    // Perform SM70 pipelining.

    // Prefetch.
    // Copy from global memory to shared memory for tile_idx_k = 0.
    cute::clear(thread_layout_A_register_tensor_A);
    cute::clear(thread_layout_B_register_tensor_B);
    // Need predicates for bounds checking.
    cute::copy_if(
        copy_A, thread_layout_A_predicate_tensor_A,
        thread_layout_A_global_block_tensor_A(cute::_, cute::_, cute::_, 0),
        thread_layout_A_register_tensor_A);
    cute::copy_if(
        copy_B, thread_layout_B_predicate_tensor_B,
        thread_layout_B_global_block_tensor_B(cute::_, cute::_, cute::_, 0),
        thread_layout_B_register_tensor_B);
    // Prepare the shared memory for the first tile iteration.
    // Copy from register to shared memory for tile_idx_k = 0.
    // No need for predicates and clear for shared memory anymore.
    cute::copy(thread_layout_A_register_tensor_A,
               thread_layout_A_smem_tensor_A);
    cute::copy(thread_layout_B_register_tensor_B,
               thread_layout_B_smem_tensor_B);
    // Synchronize to ensure the data on shared memory is ready for mma.
    __syncthreads();

    // Perform the gemm computation loop.
    auto const num_tiles_k{cute::size<2>(global_block_tensor_A)}; // k
    constexpr auto num_mmas_per_tile_k{
        cute::size<2>(thread_layout_C_register_tensor_A)}; // MMA_K
    // Prepare the registers for the first mma iteration.
    cute::copy(thread_layout_C_smem_tensor_A(cute::_, cute::_, 0),
               thread_layout_C_register_tensor_A(cute::_, cute::_, 0));
    cute::copy(thread_layout_C_smem_tensor_B(cute::_, cute::_, 0),
               thread_layout_C_register_tensor_B(cute::_, cute::_, 0));

    // We want to overlap data copy and compute.
    // The thread performs mma iteration inside the tile iteration.
    // 1. At the beginning of each tile iteration (the first mma iteration of
    // each tile iteration), the thread copies data from global memory to
    // register for the next tile iteration, asynchronously, and this can take a
    // while.
    // 2. In each mma iteration, the thread copy data from shared memory to
    // register for the next mma iteration, and perform mma for the current mma
    // iteration, asynchronously.
    // 3. Before the last mma iteration of each tile iteration, hopefully the
    // data copy from global memory to register in #1 is completed,
    // synchronization is performed. The thread copies data from the register to
    // shared memory, asynchronously. Synchronization is performed again to
    // ensure the data for the mma of the new tile iteration is ready.

    for (auto tile_idx_k{0}; tile_idx_k < num_tiles_k; ++tile_idx_k)
    {
        CUTE_UNROLL
        for (auto mma_idx_k{0}; mma_idx_k < num_mmas_per_tile_k; ++mma_idx_k)
        {
            // Before the last mma iteration of each tile iteration, copy data
            // from register to shared memory for the next tile iteration.
            if (mma_idx_k == num_mmas_per_tile_k - 1)
            {
                // Ensure the data copy from global memory to register is
                // completed.
                __syncthreads();
                // Copy data from register to shared memory for the next tile
                // iteration.
                cute::copy(thread_layout_A_register_tensor_A,
                           thread_layout_A_smem_tensor_A);
                cute::copy(thread_layout_B_register_tensor_B,
                           thread_layout_B_smem_tensor_B);
                // Ensure the data for the mma of the new tile iteration is
                // ready.
                __syncthreads();
            }
            // Copy data from shared memory to register for the next mma
            // iteration.
            auto const mma_idx_k_next{(mma_idx_k + 1) % num_mmas_per_tile_k};
            cute::copy(
                thread_layout_C_smem_tensor_A(cute::_, cute::_, mma_idx_k_next),
                thread_layout_C_register_tensor_A(cute::_, cute::_,
                                                  mma_idx_k_next));
            cute::copy(
                thread_layout_C_smem_tensor_B(cute::_, cute::_, mma_idx_k_next),
                thread_layout_C_register_tensor_B(cute::_, cute::_,
                                                  mma_idx_k_next));
            // Before the first mma iteration of each tile iteration, copy data
            // from global memory to register for the next tile iteration.
            if (mma_idx_k == 0)
            {
                auto const tile_idx_k_next{(tile_idx_k + 1) % num_tiles_k};
                cute::clear(thread_layout_A_register_tensor_A);
                cute::clear(thread_layout_B_register_tensor_B);
                // Need predicates for bounds checking.
                cute::copy_if(copy_A, thread_layout_A_predicate_tensor_A,
                              thread_layout_A_global_block_tensor_A(
                                  cute::_, cute::_, cute::_, tile_idx_k_next),
                              thread_layout_A_register_tensor_A);
                cute::copy_if(copy_B, thread_layout_B_predicate_tensor_B,
                              thread_layout_B_global_block_tensor_B(
                                  cute::_, cute::_, cute::_, tile_idx_k_next),
                              thread_layout_B_register_tensor_B);
            }
            // Perform mma for the current mma iteration.
            cute::gemm(
                mma,
                thread_layout_C_register_tensor_A(cute::_, cute::_, mma_idx_k),
                thread_layout_C_register_tensor_B(cute::_, cute::_, mma_idx_k),
                thread_layout_C_register_tensor_C);
        }
    }

    // Scale and accumulate the result from the register tensor to the global
    // block tensor.
    // There does not seem to be a tiled axpby existing yet.
    cute::axpby(alpha, thread_layout_C_register_tensor_C, beta,
                thread_layout_C_global_block_tensor_C,
                thread_layout_C_predicate_tensor_C);
}

#endif // CUT_GENERAL_MATRIX_MULTIPLICATION_CUH