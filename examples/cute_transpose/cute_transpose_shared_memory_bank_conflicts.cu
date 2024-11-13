#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_transpose.cuh"
#include "cute_transpose.hpp"

template <class TENSOR_SRC, class TENSOR_DST, class SHARED_MEMORY_LAYOUT_SRC, class SHARED_MEMORY_LAYOUT_DST, class THREAD_LAYOUT>
__global__ void transpose_naive_shared_memory_bank_conflicts(
    TENSOR_SRC tensor_src, TENSOR_DST tensor_dst, SHARED_MEMORY_LAYOUT_SRC, SHARED_MEMORY_LAYOUT_DST, THREAD_LAYOUT)
{
    using Element = typename TENSOR_SRC::value_type;
    CUTE_STATIC_ASSERT(cute::size(SHARED_MEMORY_LAYOUT_SRC{}) == cute::size(SHARED_MEMORY_LAYOUT_DST{}),
                       "SHARED_MEMORY_LAYOUT_SRC and SHARED_MEMORY_LAYOUT_DST must have the same size.");
    __shared__ Element shared_memory[cute::size(SHARED_MEMORY_LAYOUT_SRC{})];

    auto tensor_cache_src{cute::make_tensor(cute::make_smem_ptr(shared_memory), SHARED_MEMORY_LAYOUT_SRC{})};
    auto tensor_cache_dst{cute::make_tensor(cute::make_smem_ptr(shared_memory), SHARED_MEMORY_LAYOUT_DST{})};

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)
    auto global_tile_dst{
        tensor_dst(cute::make_coord(cute::_, cute::_), blockIdx.x,
                              blockIdx.y)}; // (TILE_SIZE_Y, TILE_SIZE_X)

    auto thread_global_tile_src{cute::local_partition(
        global_tile_src, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
    auto thread_global_tile_dst{cute::local_partition(
        global_tile_dst, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)

    auto thread_shared_tile_src{cute::local_partition(
        tensor_cache_src, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
    auto thread_shared_tile_dst{cute::local_partition(
        tensor_cache_dst, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)

    // A 2D array of tuples that maps (x, y) to (x, y).
    auto const identity_tensor{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_src), cute::size<1>(global_tile_src)))};
    auto const thread_identity_tensor{
        cute::local_partition(identity_tensor, THREAD_LAYOUT{}, threadIdx.x)};
    auto fragment{cute::make_tensor_like(thread_global_tile_src)};
    auto predicator{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(fragment), cute::size<1>(fragment)))};
    // auto predicator_transposed{cute::make_tensor<bool>(
    //     cute::make_shape(cute::size<1>(fragment), cute::size<0>(fragment)))};
    // printf("predicator shape (%d, %d), predicator_transposed shape (%d, %d)\n", int(cute::size<0>(predicator)), int(cute::size<1>(predicator)), int(cute::size<0>(predicator_transposed)), int(cute::size<1>(predicator_transposed)));

    auto const identity_tensor_dst{cute::make_identity_tensor(cute::make_shape(
        cute::size<0>(global_tile_dst), cute::size<1>(global_tile_dst)))};
    auto const thread_identity_tensor_dst{
        cute::local_partition(identity_tensor_dst, THREAD_LAYOUT{}, threadIdx.x)};
    auto fragment_dst{cute::make_tensor_like(thread_global_tile_dst)};
    auto predicator_dst{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(fragment_dst), cute::size<1>(fragment_dst)))};

    int const num_max_columns{cute::stride<0>(global_tile_src)};
    int const num_max_rows{cute::stride<0>(global_tile_dst)};
    constexpr int global_tile_columns{cute::size<1>(global_tile_src)};
    constexpr int global_tile_rows{cute::size<0>(global_tile_src)};
    // printf("num_max_columns: %d, num_max_rows: %d, global_tile_columns: %d, global_tile_rows: %d\n", num_max_columns, num_max_rows, global_tile_columns, global_tile_rows);

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
            // predicator(i, j) = is_row_in_bound && is_column_in_bound;
            // predicator_transposed(j, i) = is_row_in_bound && is_column_in_bound;
            // printf("blockIdx.y: %d, blockIdx.x: %d, threadIdx.x: %d, threadIdx.y: %d, i: %d, j: %d, thread_identity: (%d, %d), is_row_in_bound: %d, is_column_in_bound: %d\n", blockIdx.y, blockIdx.x, threadIdx.x, threadIdx.y, i, j, cute::get<0>(thread_identity), cute::get<1>(thread_identity), is_row_in_bound, is_column_in_bound);
            predicator(i, j) = is_row_in_bound && is_column_in_bound;
            // predicator_transposed(j, i) = is_row_in_bound && is_column_in_bound;
        }
    }

    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator_dst); ++i)
    {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator_dst); ++j)
        {
            auto const thread_identity{thread_identity_tensor_dst(i, j)};
            bool const is_row_in_bound{cute::get<0>(thread_identity) +
                                           blockIdx.x * global_tile_columns <
                                       num_max_columns};
            bool const is_column_in_bound{cute::get<1>(thread_identity) +
                                              blockIdx.y * global_tile_rows <
                                          num_max_rows};
            // predicator(i, j) = is_row_in_bound && is_column_in_bound;
            // predicator_transposed(j, i) = is_row_in_bound && is_column_in_bound;
            // printf("blockIdx.y: %d, blockIdx.x: %d, threadIdx.x: %d, threadIdx.y: %d, i: %d, j: %d, thread_identity: (%d, %d), is_row_in_bound: %d, is_column_in_bound: %d\n", blockIdx.y, blockIdx.x, threadIdx.x, threadIdx.y, i, j, cute::get<0>(thread_identity), cute::get<1>(thread_identity), is_row_in_bound, is_column_in_bound);
            predicator_dst(i, j) = is_row_in_bound && is_column_in_bound;
            // predicator_transposed(j, i) = is_row_in_bound && is_column_in_bound;
        }
    }


    cute::copy_if(predicator, thread_global_tile_src, thread_shared_tile_src);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    // // Print values from the shared memory.
    // if (threadIdx.x == 0 && threadIdx.y == 0)
    // {
    //     for (unsigned int i{0}; i < 8; ++i)
    //     {
    //         printf("shared memory[%d]: %f\n", i, shared_memory[i]);
    //     }
    // }
    // cute::copy_if(predicator_transposed, thread_shared_tile_dst, thread_global_tile_dst);
    for (unsigned int i{0}; i < cute::size(thread_shared_tile_dst); ++i)
    {
        // printf("blockIdx.y: %d, blockIdx.x: %d, threadIdx.x: %d, threadIdx.y: %d, i: %d, thread_shared_tile_dst: %f, predicate_transposed: %d\n", blockIdx.y, blockIdx.x, threadIdx.x, threadIdx.y, i, thread_shared_tile_dst(i), predicator_transposed(i) ? 1 : 0);
        if (predicator_dst(i))
        {
            thread_global_tile_dst(i) = thread_shared_tile_dst(i);
        }
    }

    // cute::copy(thread_global_tile_src, thread_shared_tile_src);
    // cute::cp_async_fence();
    // cute::cp_async_wait<0>();
    // __syncthreads();
    // cute::copy(thread_shared_tile_dst, thread_global_tile_dst);
}

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflicts(T const* input_matrix,
                                                          T* output_matrix,
                                                          unsigned int M,
                                                          unsigned int N,
                                                          cudaStream_t stream)
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

    using TILE_SIZE_X = cute::Int<64>; // bN
    using TILE_SIZE_Y = cute::Int<32>; // bM

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};
    constexpr auto block_shape_transposed{cute::make_shape(TILE_SIZE_X{}, TILE_SIZE_Y{})};

    auto const shared_memory_layout_src{cute::make_layout(
        block_shape, cute::GenRowMajor{})}; // (bM, bN) : (bN, 1)
    auto const shared_memory_layout_dst{cute::make_layout(
        block_shape_transposed, cute::GenRowMajor{})}; // (bN, bM) : (bM, 1)
    auto const shared_memory_layout_dst_transposed{cute::make_layout(
        block_shape, cute::GenColMajor{})}; // (bM, bN) : (1, bM)

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M /
                                   // TILE_SIZE_Y, N / TILE_SIZE_X)
    // cute::print(tiled_tensor_src);
    // std::cout << std::endl;
    auto const tiled_tensor_dst{cute::tiled_divide(
        tensor_dst, block_shape_transposed)}; // ((TILE_SIZE_X, TILE_SIZE_Y), N
                                              // / TILE_SIZE_X, M / TILE_SIZE_Y)
    // cute::print(tiled_tensor_dst);
    // std::cout << std::endl;
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M
                                              // / TILE_SIZE_Y, N / TILE_SIZE_X)
    // cute::print(tiled_tensor_dst_transposed);
    // std::cout << std::endl;

    using THREAD_BLOCK_SIZE_X = cute::Int<32>; // tN
    using THREAD_BLOCK_SIZE_Y = cute::Int<8>;  // tM

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    CUTE_STATIC_ASSERT(TILE_SIZE_Y::value % THREAD_BLOCK_SIZE_Y::value == 0,
                       "TILE_SIZE_Y must be divisible by THREAD_BLOCK_SIZE_Y");

    constexpr auto thread_block_shape{
        cute::make_shape(THREAD_BLOCK_SIZE_Y{}, THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};

    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    transpose_naive_shared_memory_bank_conflicts<<<grid_dim, thread_dim, 0,
                                                   stream>>>(
        tiled_tensor_src, tiled_tensor_dst, shared_memory_layout_dst_transposed, shared_memory_layout_dst, thread_layout);

    return cudaGetLastError();
}

// Explicit instantiation.
template cudaError_t launch_transpose_shared_memory_bank_conflicts<float>(
    float const* input_matrix, float* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_transpose_shared_memory_bank_conflicts<double>(
    double const* input_matrix, double* output_matrix, unsigned int M,
    unsigned int N, cudaStream_t stream);
template cudaError_t launch_transpose_shared_memory_bank_conflicts<int>(
    int const* input_matrix, int* output_matrix, unsigned int M, unsigned int N,
    cudaStream_t stream);
template cudaError_t
launch_transpose_shared_memory_bank_conflicts<unsigned int>(
    unsigned int const* input_matrix, unsigned int* output_matrix,
    unsigned int M, unsigned int N, cudaStream_t stream);
