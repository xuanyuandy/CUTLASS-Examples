#include <iostream>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

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

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_naive(TENSOR_SRC tensor_src,
                                TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT)
{
    using Element = typename TENSOR_SRC::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_),
                                    blockIdx.y,
                                    blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)
    auto global_tile_dst_transposed{
        tensor_dst_transposed(cute::make_coord(cute::_, cute::_), blockIdx.y,
                              blockIdx.x)}; // (TILE_SIZE_Y, TILE_SIZE_X)

    auto thread_tile_src{cute::local_partition(
        global_tile_src, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)
    auto thread_tile_dst_transposed{cute::local_partition(
        global_tile_dst_transposed, THREAD_LAYOUT{},
        threadIdx.x)}; // (THREAD_VALUE_SIZE_Y, THREAD_VALUE_SIZE_X)

    // A 2D array of tuples that maps (x, y) to (x, y).
    auto const identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_tile_src),
                         cute::size<1>(global_tile_src)))};
    auto const thread_identity_tensor{cute::local_partition(
        identity_tensor, THREAD_LAYOUT{}, threadIdx.x)};
    auto fragment{cute::make_tensor_like(thread_tile_src)};
    auto predicator{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(fragment),
                         cute::size<1>(fragment)))};

    // These might not always be the size of the original intact tensor.
    // The user should provide the information instead.
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
            bool const is_row_in_bound{cute::get<0>(thread_identity) + blockIdx.y * global_tile_rows < num_max_rows};
            bool const is_column_in_bound{cute::get<1>(thread_identity) + blockIdx.x * global_tile_columns < num_max_columns};
            predicator(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    cute::copy_if(predicator, thread_tile_src, fragment);
    cute::copy_if(predicator, fragment, thread_tile_dst_transposed);

    // Alternatively, we could just do the following instead.
    // cute::copy_if(predicator, thread_tile_src, thread_tile_dst_transposed);
}

// Transpose a M x N row-major matrix.
template <class T>
void transpose(T const* src, T* dst, unsigned int M, unsigned int N)
{
    for (unsigned int i{0}; i < M; ++i)
    {
        for (unsigned int j{0}; j < N; ++j)
        {
            dst[j * M + i] = src[i * N + j];
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

// Compare a data array with a reference array.
template <class T>
bool compare(T const* data, T const* ref, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        if (data[i] != ref[i])
        {
            std::cout << i << " " << data[i] << " " << ref[i] << std::endl;
            return false;
        }
    }

    return true;
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

int main()
{
    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    using Element = int;

    unsigned int const M{74}; // Number of columns.
    unsigned int const N{13}; // Number of rows.

    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    thrust::host_vector<Element> h_src(cute::size(tensor_shape));
    thrust::host_vector<Element> h_dst(cute::size(tensor_shape_transposed));
    thrust::host_vector<Element> h_dst_ref(cute::size(tensor_shape_transposed));

    initialize(h_src.data(), h_src.size());
    transpose(h_src.data(), h_dst_ref.data(), M, N);

    thrust::device_vector<Element> d_src{h_src};
    thrust::device_vector<Element> d_dst{h_dst};

    // Input matrix: row-major M x N matrix.
    auto const global_memory_layout_src{cute::make_layout(
        tensor_shape, cute::GenRowMajor{})}; // (M, N) : (N, 1)
    // Output matrix: row-major N x M matrix.
    auto const global_memory_layout_dst{cute::make_layout(
        tensor_shape_transposed, cute::GenRowMajor{})}; // (N, M) : (M, 1)
    // Same output matrix, but different view: column-major M x N matrix.
    auto const global_memory_layout_dst_transposed{cute::make_layout(
        tensor_shape, cute::GenColMajor{})}; // (M, N) : (1, M)

    auto const tensor_src{cute::make_tensor(
        cute::make_gmem_ptr(thrust::raw_pointer_cast(d_src.data())),
        global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(
        cute::make_gmem_ptr(thrust::raw_pointer_cast(d_dst.data())),
        global_memory_layout_dst)};
    auto const tensor_dst_transposed{cute::make_tensor(
        cute::make_gmem_ptr(thrust::raw_pointer_cast(d_dst.data())),
        global_memory_layout_dst_transposed)};

    using TILE_SIZE_X = cute::Int<64>;
    using TILE_SIZE_Y = cute::Int<32>;

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M /
                                   // TILE_SIZE_Y, N / TILE_SIZE_X)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), M
                                              // / TILE_SIZE_Y, N / TILE_SIZE_X)

    auto const g_src_example{
        tiled_tensor_src(cute::make_coord(cute::_, cute::_), 1, 2)};
    auto const g_dst_example{
        tiled_tensor_dst_transposed(cute::make_coord(cute::_, cute::_), 1, 2)};

    using THREAD_BLOCK_SIZE_X = cute::Int<32>;
    using THREAD_BLOCK_SIZE_Y = cute::Int<8>;

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

    transpose_naive<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed, thread_layout);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy the data from device to host.
    h_dst = d_dst;

    if (compare(h_dst.data(), h_dst_ref.data(), h_dst.size()))
    {
        std::cout << "Success!" << std::endl;
    }
    else
    {
        std::cout << "Failure!" << std::endl;
    }

    // print(h_dst.data(), h_dst_ref.data(), h_dst.size());
}