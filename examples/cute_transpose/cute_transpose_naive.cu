#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
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
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT>
__global__ void transpose_naive(TENSOR_SRC tensor_src, TENSOR_DST tensor_dst_transposed, THREAD_LAYOUT)
{
    using Element = typename TENSOR_SRC::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y)}; // (TILE_SIZE_X, TILE_SIZE_Y)
    auto global_tile_dst_transposed{tensor_dst_transposed(cute::make_coord(cute::_, cute::_), blockIdx.x, blockIdx.y)}; // (TILE_SIZE_Y, TILE_SIZE_X)

    auto thread_tile_src{cute::local_partition(global_tile_src, THREAD_LAYOUT{}, threadIdx.x)}; // (THREAD_VALUE_SIZE_X, THREAD_VALUE_SIZE_Y)
    auto thread_tile_dst_transposed{cute::local_partition(global_tile_dst_transposed, THREAD_LAYOUT{}, threadIdx.x)}; // (THREAD_VALUE_SIZE_X, THREAD_VALUE_SIZE_Y)

    auto register_fragment{cute::make_tensor_like(thread_tile_src)};

    cute::copy(thread_tile_src, register_fragment);
    cute::copy(register_fragment, thread_tile_dst_transposed);
}

template <class T>
void transpose(T const* src, T* dst, unsigned int M, unsigned int N)
{
    for (unsigned int i{0}; i < N; ++i)
    {
        for (unsigned int j{0}; j < M; ++j)
        {
            dst[j * N + i] = src[i * M + j];
        }
    }
}

template <class T>
void initialize(T* data, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        data[i] = static_cast<T>(i);
    }
}

template <class T>
bool compare(T const* data, T const* ref, unsigned int size)
{
    for (unsigned int i{0}; i < size; ++i)
    {
        // std::cout << i << " " << data[i] << " " << ref[i] << std::endl;
        if (data[i] != ref[i])
        {
            std::cout << i << " " << data[i] << " " << ref[i] << std::endl;
            return false;
        }
    }

    return true;
}

int main()
{
    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    using Element = int;

    // unsigned int const M{2048};
    // unsigned int const N{512};

    unsigned int const M{8192};
    unsigned int const N{8192};

    auto const tensor_shape{cute::make_shape(M, N)};
    auto const tensor_shape_transposed{cute::make_shape(N, M)};

    thrust::host_vector<Element> h_src(cute::size(tensor_shape));
    thrust::host_vector<Element> h_dst(cute::size(tensor_shape_transposed));
    thrust::host_vector<Element> h_dst_ref(cute::size(tensor_shape_transposed));

    initialize(h_src.data(), h_src.size());
    transpose(h_src.data(), h_dst_ref.data(), M, N);

    thrust::device_vector<Element> d_src{h_src};
    thrust::device_vector<Element> d_dst{h_dst};

    auto const global_memory_layout_src{cute::make_layout(tensor_shape, cute::GenRowMajor{})};
    auto const global_memory_layout_dst{cute::make_layout(tensor_shape_transposed, cute::GenRowMajor{})};
    auto const global_memory_layout_dst_transposed{cute::make_layout(tensor_shape, cute::GenColMajor{})};

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(thrust::raw_pointer_cast(d_src.data())), global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(thrust::raw_pointer_cast(d_dst.data())), global_memory_layout_dst)};
    auto const tensor_dst_transposed{cute::make_tensor(cute::make_gmem_ptr(thrust::raw_pointer_cast(d_dst.data())), global_memory_layout_dst_transposed)};


    using TILE_SIZE_X = cute::Int<64>;
    using TILE_SIZE_Y = cute::Int<64>;

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_X{}, TILE_SIZE_Y{})};
    constexpr auto block_shape_transposed{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};

    auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)}; // ((TILE_SIZE_X, TILE_SIZE_Y), M / TILE_SIZE_X, N / TILE_SIZE_Y)
    cute::print(tiled_tensor_src);
    std::cout << std::endl;
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), N / TILE_SIZE_Y, M / TILE_SIZE_X)
    // auto const tiled_tensor_dst_transposed{cute::tiled_divide(tensor_dst_transposed, block_shape_transposed)}; // ((TILE_SIZE_Y, TILE_SIZE_X), N / TILE_SIZE_Y, M / TILE_SIZE_X)
    cute::print(tiled_tensor_dst_transposed);
    std::cout << std::endl;

    using THREAD_BLOCK_SIZE_X = cute::Int<8>;
    using THREAD_BLOCK_SIZE_Y = cute::Int<32>;

    constexpr auto thread_block_shape{cute::make_shape(THREAD_BLOCK_SIZE_X{}, THREAD_BLOCK_SIZE_Y{})};
    constexpr auto thread_layout{cute::make_layout(thread_block_shape, cute::GenRowMajor{})};

    dim3 const grid_dim{cute::size<1>(tiled_tensor_src), cute::size<2>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    transpose_naive<<<grid_dim, thread_dim, 0, stream>>>(tiled_tensor_src, tiled_tensor_dst_transposed, thread_layout);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    h_dst = d_dst;

    if (compare(h_dst.data(), h_dst_ref.data(), h_dst.size()))
    {
        std::cout << "Success!" << std::endl;
    }
    else
    {
        std::cout << "Failure!" << std::endl;
    }
}