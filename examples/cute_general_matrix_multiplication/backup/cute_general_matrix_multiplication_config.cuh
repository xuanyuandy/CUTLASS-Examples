#ifndef CUTE_GENERAL_MATRIX_MULTIPLICATION_CONFIG_CUH
#define CUTE_GENERAL_MATRIX_MULTIPLICATION_CONFIG_CUH

#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

constexpr int constexpr_log2(int n)
{
    return ((n < 2) ? 0 : 1 + constexpr_log2(n / 2));
}

template <class MmaOperation, int TileSizeM, int TileSizeN, int TileSizeK,
          int NumThreads>
struct MmaConfig
{
};

template <>
struct MmaConfig<cute::SM80_16x8x16_F16F16F16F16_TN, 128, 128, 32, 128>
{
    // Configure SM80 Tensor Core MMA.
    using MmaTraits = cute::MMA_Traits<cute::SM80_16x8x16_F16F16F16F16_TN>;
    using MmaAtomShape = MmaTraits::Shape_MNK;
    using MmaAtom = cute::MMA_Atom<MmaTraits>;

    static constexpr int MMA_LAYOUT_M{2};
    static constexpr int MMA_LAYOUT_N{2};
    static constexpr int MMA_LAYOUT_K{1};
    static constexpr auto mma_layout{cute::make_layout(
        cute::make_shape(cute::Int<MMA_LAYOUT_M>{}, cute::Int<MMA_LAYOUT_N>{},
                         cute::Int<MMA_LAYOUT_K>{}))};
    static constexpr int NUM_MMA_TILE_M{1};
    static constexpr int NUM_MMA_TILE_N{2};
    static constexpr int NUM_MMA_TILE_K{1};
    static constexpr int MMA_TILE_M{cute::get<0>(MmaAtomShape{}) *
                                    MMA_LAYOUT_M * NUM_MMA_TILE_M};
    static constexpr int MMA_TILE_N{cute::get<1>(MmaAtomShape{}) *
                                    MMA_LAYOUT_N * NUM_MMA_TILE_N};
    static constexpr int MMA_TILE_K{cute::get<2>(MmaAtomShape{}) *
                                    MMA_LAYOUT_K * NUM_MMA_TILE_K};
    static constexpr auto mma_tile{cute::make_tile(cute::Int<MMA_TILE_M>{},
                                                   cute::Int<MMA_TILE_N>{},
                                                   cute::Int<MMA_TILE_K>{})};
    static constexpr auto mma{
        cute::make_tiled_mma(MmaAtom{}, mma_layout, mma_tile)};

    CUTE_STATIC_ASSERT_V(cute::size(mma) == cute::Int<128>{});

    CUTE_STATIC_ASSERT_V(cute::Int<128>{} % cute::size<0>(mma_tile) ==
                         cute::Int<0>{}); // BLK_M % MMA_TILE_M == 0
    CUTE_STATIC_ASSERT_V(cute::Int<128>{} % cute::size<1>(mma_tile) ==
                         cute::Int<0>{}); // BLK_N % MMA_TILE_N == 0
    CUTE_STATIC_ASSERT_V(cute::Int<32>{} % cute::size<2>(mma_tile) ==
                         cute::Int<0>{}); // BLK_K % MMA_TILE_K == 0
};

template <class TA, class TB, class TC, class VectorTypeA, class VectorTypeB,
          int TileSizeM, int TileSizeN, int TileSizeK, int NumThreads>
struct MemoryConfig
{
    // Define CTA size.
    static constexpr auto const bM{cute::Int<TileSizeM>{}};
    static constexpr auto const bN{cute::Int<TileSizeN>{}};
    static constexpr auto const bK{cute::Int<TileSizeK>{}};
    static constexpr auto const cta_tiler{
        cute::make_shape(bM, bN, bK)}; // (BLK_M, BLK_N, BLK_K)

    // Define smem layouts.
    // smem_layout_A is (BLK_M, BLK_K) column-major.
    // smem_layout_B is (BLK_N, BLK_K) column-major.
    // smem_layout_C is (BLK_M, BLK_N) column-major.
    static constexpr auto const smem_shape_A{
        cute::make_shape(bM, bK)}; // (BLK_M, BLK_K)
    static constexpr auto const smem_stride_A{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(smem_shape_A))}; // column-major
    static constexpr auto const smem_layout_A{
        cute::make_layout(smem_shape_A, smem_stride_A)}; // (BLK_M, BLK_K)
    static constexpr auto const smem_shape_B{
        cute::make_shape(bN, bK)}; // (BLK_N, BLK_K)
    static constexpr auto const smem_stride_B{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(smem_shape_B))}; // column-major
    static constexpr auto const smem_layout_B{
        cute::make_layout(smem_shape_B, smem_stride_B)}; // (BLK_N, BLK_K)
    static constexpr auto const smem_shape_C{
        cute::make_shape(bM, bN)}; // (BLK_M, BLK_N)
    static constexpr auto const smem_stride_C{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(smem_shape_C))}; // column-major
    static constexpr auto const smem_layout_C{
        cute::make_layout(smem_shape_C, smem_stride_C)}; // (BLK_M, BLK_N)

    // Define thread layouts.
    static constexpr auto const thread_shape_A{
        cute::make_shape(cute::Int<16>{}, cute::Int<8>{})}; // (THR_M, THR_K)
    static constexpr auto const thread_shape_B{
        cute::make_shape(cute::Int<16>{}, cute::Int<8>{})}; // (THR_N, THR_K)
    static constexpr auto const thread_shape_C{
        cute::make_shape(cute::Int<32>{}, cute::Int<4>{})}; // (THR_M, THR_N)
    static constexpr auto const thread_stride_A{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(thread_shape_A))}; // column-major
    static constexpr auto const thread_stride_B{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(thread_shape_B))}; // column-major
    static constexpr auto const thread_stride_C{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(thread_shape_C))}; // column-major
    static constexpr auto const thread_layout_A{
        cute::make_layout(thread_shape_A, thread_stride_A)}; // (THR_M, THR_K)
    static constexpr auto const thread_layout_B{
        cute::make_layout(thread_shape_B, thread_stride_B)}; // (THR_N, THR_K)
    static constexpr auto const thread_layout_C{
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

    static constexpr auto NUM_VECTOR_ELEMENTS_A{sizeof(VectorTypeA) /
                                                sizeof(TA)};
    static constexpr auto const vector_shape_A{
        cute::make_shape(cute::Int<NUM_VECTOR_ELEMENTS_A>{},
                         cute::Int<1>{})}; // (NUM_VECTOR_ELEMENTS_A, 1)
    static constexpr auto const vector_stride_A{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(vector_shape_A))}; // column-major
    static constexpr auto const vector_layout_A{cute::make_layout(
        vector_shape_A, vector_stride_A)}; // (NUM_VECTOR_ELEMENTS_A, 1)
    TiledCopy const copy_A{cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<VectorTypeA>, TA>{},
        thread_layout_A,
        vector_layout_A)}; // Thread layout: (THR_M, THR_K) Value layout:
                           // (NUM_VECTOR_ELEMENTS_A, 1)
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(smem_layout_A) %
            (cute::size<0>(thread_layout_A) * cute::size<0>(vector_layout_A)) ==
        cute::Int<0>{}); // BLK_M % (THR_M * NUM_VECTOR_ELEMENTS_A) == 0

    static constexpr auto NUM_VECTOR_ELEMENTS_B{sizeof(VectorTypeB) /
                                                sizeof(TB)};
    TiledCopy const vector_shape_B{
        cute::make_shape(cute::Int<NUM_VECTOR_ELEMENTS_B>{},
                         cute::Int<1>{})}; // (NUM_VECTOR_ELEMENTS_B, 1)
    static constexpr auto const vector_stride_B{cute::make_stride(
        cute::Int<1>{}, cute::size<0>(vector_shape_B))}; // column-major
    static constexpr auto const vector_layout_B{cute::make_layout(
        vector_shape_B, vector_stride_B)}; // (NUM_VECTOR_ELEMENTS_B, 1)
    static constexpr auto const copy_B{cute::make_tiled_copy(
        cute::Copy_Atom<cute::UniversalCopy<VectorTypeB>, TB>{},
        thread_layout_B,
        vector_layout_B)}; // Thread layout: (THR_N, THR_K) Value layout:
                           // (NUM_VECTOR_ELEMENTS_B, 1)
    CUTE_STATIC_ASSERT_V(
        cute::size<0>(smem_layout_B) %
            (cute::size<0>(thread_layout_B) * cute::size<0>(vector_layout_B)) ==
        cute::Int<0>{}); // BLK_N % (THR_N * NUM_VECTOR_ELEMENTS_B) == 0

    // Swizzle parameters.
    static constexpr int NUM_BASE_BITS_A{constexpr_log2(NUM_VECTOR_ELEMENTS_A)};
    static constexpr int NUM_MASK_BITS_A{constexpr_log2(32 * 4 / sizeof(TA)) -
                                         NUM_BASE_BITS_A};
    static constexpr int NUM_SHIFT_BITS_A{constexpr_log2(bM) - NUM_BASE_BITS_A};

    static constexpr int NUM_BASE_BITS_B{constexpr_log2(NUM_VECTOR_ELEMENTS_B)};
    static constexpr int NUM_MASK_BITS_B{constexpr_log2(32 * 4 / sizeof(TB)) -
                                         NUM_BASE_BITS_B};
    static constexpr int NUM_SHIFT_BITS_B{constexpr_log2(bN) - NUM_BASE_BITS_B};

    static constexpr auto const swizzle_A{
        cute::Swizzle<NUM_MASK_BITS_A, NUM_BASE_BITS_A, NUM_SHIFT_BITS_A>{}};
    static constexpr auto const swizzle_B{
        cute::Swizzle<NUM_MASK_BITS_B, NUM_BASE_BITS_B, NUM_SHIFT_BITS_B>{}};

    // In fact, for some layouts, swizzles are not needed if no transpose is
    // performed.
    // But it should not reduce the performance even if the transpose is not
    // performed.
    static constexpr auto const smem_layout_A_swizzled{
        cute::composition(swizzle_A, smem_layout_A)};
    static constexpr auto const smem_layout_B_swizzled{
        cute::composition(swizzle_B, smem_layout_B)};
};

#endif // CUT_GENERAL_MATRIX_MULTIPLICATION_CONFIG_CUH