#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_matrix_transpose.hpp"

template <class TensorSrc, class TensorDst, class ThreadLayout>
static __global__ void matrix_transpose_naive(TensorSrc tensor_src,
                                              TensorDst tensor_dst_transposed,
                                              ThreadLayout)
{
    using Element = typename TensorSrc::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TileSizeY, TileSizeX)
    auto global_tile_dst_transposed{
        tensor_dst_transposed(cute::make_coord(cute::_, cute::_), blockIdx.y,
                              blockIdx.x)}; // (TileSizeY, TileSizeX)

    auto thread_global_tile_src{cute::local_partition(
        global_tile_src, ThreadLayout{},
        threadIdx.x)}; // (ThreadValueSizeY, ThreadValueSizeX)
    auto thread_global_tile_dst_transposed{cute::local_partition(
        global_tile_dst_transposed, ThreadLayout{},
        threadIdx.x)}; // (ThreadValueSizeY, ThreadValueSizeX)

    // A 2D array of tuples that maps (x, y) to (x, y).
    auto const identity_tensor{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_src), cute::size<1>(global_tile_src)))};
    auto const thread_identity_tensor{
        cute::local_partition(identity_tensor, ThreadLayout{}, threadIdx.x)};
    auto fragment{cute::make_tensor_like(thread_global_tile_src)};
    auto predicator{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(fragment), cute::size<1>(fragment)))};

    auto const num_max_columns{cute::stride<0>(global_tile_src)};
    auto const num_max_rows{cute::stride<1>(global_tile_dst_transposed)};
    constexpr auto global_tile_columns{cute::size<1>(global_tile_src)};
    constexpr auto global_tile_rows{cute::size<0>(global_tile_src)};

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator); ++i)
    {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator); ++j)
        {
            auto const thread_identity{thread_identity_tensor(i, j)};
            bool const is_row_in_bound{cute::get<0>(thread_identity) +
                                           blockIdx.y * global_tile_rows <
                                       num_max_rows};
            bool const is_column_in_bound{cute::get<1>(thread_identity) +
                                              blockIdx.x * global_tile_columns <
                                          num_max_columns};
            predicator(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    cute::copy_if(predicator, thread_global_tile_src, fragment);
    cute::copy_if(predicator, fragment, thread_global_tile_dst_transposed);

    // Alternatively, we could just do the following instead.
    // cute::copy_if(predicator, thread_global_tile_src,
    // thread_global_tile_dst_transposed);
}

enum class GlobalMemoryCoalescedAccessMode
{
    Read,
    Write
};

template <typename T>
static cudaError_t launch_matrix_transpose_naive_base(
    T const* input_matrix, T* output_matrix, unsigned int M, unsigned int N,
    GlobalMemoryCoalescedAccessMode coalesced_access_mode, cudaStream_t stream)
{
    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    // Input matrix: row-major M x N matrix.
    auto const global_memory_layout_src{cute::make_layout(
        tensor_shape, cute::GenRowMajor{})}; // (M, N) : (N, 1)
    // Output matrix: row-major N x M matrix.
    auto const global_memory_layout_dst{cute::make_layout(
        tensor_shape_transposed, cute::GenRowMajor{})}; // (N, M) : (M, 1)
    // Same output matrix, but different view: column-major M x N matrix.
    auto const global_memory_layout_dst_transposed{cute::make_layout(
        tensor_shape, cute::GenColMajor{})}; // (M, N) : (1, M)

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_matrix),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                                            global_memory_layout_dst)};
    auto const tensor_dst_transposed{
        cute::make_tensor(cute::make_gmem_ptr(output_matrix),
                          global_memory_layout_dst_transposed)};

    using TileSizeX = cute::Int<64>; // bN
    using TileSizeY = cute::Int<32>; // bM

    constexpr auto block_shape{cute::make_shape(TileSizeY{}, TileSizeX{})};

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TileSizeY, TileSizeX), M /
                                   // TileSizeY, N / TileSizeX)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TileSizeY, TileSizeX), M
                                              // / TileSizeY, N / TileSizeX)

    using ThreadBlockSizeX = cute::Int<32>; // tN
    using ThreadBlockSizeY = cute::Int<8>;  // tM

    constexpr auto thread_block_shape{
        cute::make_shape(ThreadBlockSizeY{}, ThreadBlockSizeX{})};
    constexpr auto thread_block_shape_transposed{
        cute::make_shape(ThreadBlockSizeX{}, ThreadBlockSizeY{})};
    // Coalesced memory read.
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
    // Coalesced memory write.
    constexpr auto thread_layout_transposed{
        cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{
        cute::size(ThreadBlockSizeX::value * ThreadBlockSizeY::value)};

    if (coalesced_access_mode == GlobalMemoryCoalescedAccessMode::Read)
    {
        CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeX{} == cute::Int<0>{},
                             "TileSizeX must be divisible by ThreadBlockSizeX");
        CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeY{} == cute::Int<0>{},
                             "TileSizeY must be divisible by ThreadBlockSizeY");
        matrix_transpose_naive<<<grid_dim, thread_dim, 0, stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed, thread_layout);
    }
    else
    {
        CUTE_STATIC_ASSERT_V(TileSizeX{} % ThreadBlockSizeY{} == cute::Int<0>{},
                             "TileSizeX must be divisible by ThreadBlockSizeY");
        CUTE_STATIC_ASSERT_V(TileSizeY{} % ThreadBlockSizeX{} == cute::Int<0>{},
                             "TileSizeY must be divisible by ThreadBlockSizeX");
        matrix_transpose_naive<<<grid_dim, thread_dim, 0, stream>>>(
            tiled_tensor_src, tiled_tensor_dst_transposed,
            thread_layout_transposed);
    }

    return cudaGetLastError();
}

template <typename T>
cudaError_t launch_matrix_transpose_naive_coalesced_read(T const* input_matrix,
                                                         T* output_matrix,
                                                         unsigned int M,
                                                         unsigned int N,
                                                         cudaStream_t stream)
{
    return launch_matrix_transpose_naive_base(
        input_matrix, output_matrix, M, N,
        GlobalMemoryCoalescedAccessMode::Read, stream);
}

template <typename T>
cudaError_t launch_matrix_transpose_naive_coalesced_write(T const* input_matrix,
                                                          T* output_matrix,
                                                          unsigned int M,
                                                          unsigned int N,
                                                          cudaStream_t stream)
{
    return launch_matrix_transpose_naive_base(
        input_matrix, output_matrix, M, N,
        GlobalMemoryCoalescedAccessMode::Write, stream);
}

// Explicit instantiation.
template cudaError_t launch_matrix_transpose_naive_coalesced_read<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_matrix_transpose_naive_coalesced_read<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);

template cudaError_t launch_matrix_transpose_naive_coalesced_write<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_matrix_transpose_naive_coalesced_write<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
