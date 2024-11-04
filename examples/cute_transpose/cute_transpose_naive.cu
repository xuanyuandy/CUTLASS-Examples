#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include <cuda_runtime.h>

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

    // This might be too large. Consider using 1D tensor for smaller register usage.
    auto const identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_tile_src),
                         cute::size<1>(global_tile_src)))};
    // Print identity tensor
    for (unsigned int i{0}; i < cute::size<0>(identity_tensor); ++i)
    {
        for (unsigned int j{0}; j < cute::size<1>(identity_tensor); ++j)
        {
            printf("i: %d j: %d %d %d\n", i, j, cute::get<0>(identity_tensor(i, j)), cute::get<1>(identity_tensor(i, j)));
        }
    }
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
    printf("global_tile_columns: %d\n", global_tile_columns);
    printf("global_tile_rows: %d\n", global_tile_rows);
    printf("num_max_columns: %d\n", num_max_columns);
    printf("num_max_rows: %d\n", num_max_rows);
    for (unsigned int i{0}; i < cute::size<0>(fragment); ++i)
    {
        for (unsigned int j{0}; j < cute::size<1>(fragment); ++j)
        {
            bool const is_row_in_bound{cute::get<0>(thread_identity_tensor(i, j)) + blockIdx.y * global_tile_rows < num_max_rows};
            bool const is_column_in_bound{cute::get<1>(thread_identity_tensor(i, j)) + blockIdx.x * global_tile_columns < num_max_columns};
            predicator(i, j) = is_row_in_bound && is_column_in_bound;
            // Print predicator
            printf("predicator block(%d, %d) thread(%d) %d %d is_row_in_bound %s is_column_in_bound %s predicator %s identity value <0> %d <1> %d\n", blockIdx.x, blockIdx.y, threadIdx.x, i, j, is_row_in_bound ? "true" : "false", is_column_in_bound ? "true" : "false", predicator(i, j) ? "true" : "false", cute::get<0>(thread_identity_tensor(i, j)), cute::get<1>(thread_identity_tensor(i, j)));
        }
    }

    printf("@@@@@@@@@@@@@@@@@@@@\n");


    if (blockIdx.x == 2 && blockIdx.y == 1 && threadIdx.x == 3)
    {
        printf("sdfsffs num_max_columns: %d\n", num_max_columns);
        printf("sdfsffs num_max_rows: %d\n", num_max_rows);
    }

    // constexpr auto global_tile_columns{cute::size<1>(global_tile_src)};
    // constexpr auto global_tile_rows{cute::size<0>(global_tile_src)};
    // constexpr auto global_tile_columns{8};
    // constexpr auto global_tile_rows{4};
    if (blockIdx.x == 2 && blockIdx.y == 1 && threadIdx.x == 3)
    {
        printf("global_tile_columns: %d\n", global_tile_columns);
        printf("global_tile_rows: %d\n", global_tile_rows);
    }

    constexpr auto thread_tile_columns{cute::size<1>(thread_tile_src)};
    constexpr auto thread_tile_rows{cute::size<0>(thread_tile_src)};
    if (blockIdx.x == 2 && blockIdx.y == 1 && threadIdx.x == 3)
    {
        printf("thread_tile_columns: %d\n", thread_tile_columns);
        printf("thread_tile_rows: %d\n", thread_tile_rows);
    }

    // These two should be constexpr. But how could we implement these?
    auto const num_tiles_x{cute::size<1>(global_tile_src) / thread_tile_columns};
    auto const num_tiles_y{cute::size<0>(global_tile_src) / thread_tile_rows};

    auto const thread_block_idx_x{threadIdx.x % num_tiles_x};
    auto const thread_block_idx_y{threadIdx.x / num_tiles_x};

    // auto const thread_block_idx_x{threadIdx.x / num_tiles_y};
    // auto const thread_block_idx_y{threadIdx.x % num_tiles_y};

    if (blockIdx.x == 2 && blockIdx.y == 1 && threadIdx.x == 3)
    {
        printf("thread_block_idx_x: %d\n", thread_block_idx_x);
        printf("thread_block_idx_y: %d\n", thread_block_idx_y);
    }

    auto register_fragment{cute::make_tensor_like(thread_tile_src)};
    auto register_fragment_predicator{cute::make_tensor<bool>(
        cute::make_shape(cute::size<0>(register_fragment),
                         cute::size<1>(register_fragment)))};
    auto register_fragment_identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(register_fragment),
                         cute::size<1>(register_fragment)))};
    
    // for (unsigned int i{0}; i < cute::size<0>(register_fragment_identity_tensor); ++i)
    // {
    //     for (unsigned int j{0}; j < cute::size<1>(register_fragment_identity_tensor); ++j)
    //     {
    //         if (blockIdx.x == 2 && blockIdx.y == 1 && threadIdx.x == 3)
    //         {
    //             printf("mmmmmmm %d %d %d %d\n", i, j, cute::get<0>(register_fragment_identity_tensor(i, j)), cute::get<1>(register_fragment_identity_tensor(i, j)));
    //         }
    //         register_fragment_predicator(i, j) = (cute::get<0>(register_fragment_identity_tensor(i, j)) + blockIdx.y * global_tile_rows + thread_block_idx_y * thread_tile_rows < num_max_rows) &&
    //                                             (cute::get<1>(register_fragment_identity_tensor(i, j)) + blockIdx.x * global_tile_columns + thread_block_idx_x * thread_tile_columns < num_max_columns);
    //     }
    // }

    if (blockIdx.x == 2 && blockIdx.y == 1 && threadIdx.x == 3)
    {
        printf("Printing predicator\n");
        for (unsigned int i{0}; i < cute::size<0>(register_fragment_identity_tensor); ++i)
        {
            for (unsigned int j{0}; j < cute::size<1>(register_fragment_identity_tensor); ++j)
            {
                // Print predicator
                printf("sds %d %d %s\n", i, j, register_fragment_predicator(i, j) ? "true" : "false");
            }
        }
    }

    // printf("Printing predicator\n");
    // for (unsigned int i{0}; i < cute::size<0>(register_fragment_identity_tensor); ++i)
    // {
    //     for (unsigned int j{0}; j < cute::size<1>(register_fragment_identity_tensor); ++j)
    //     {
    //         // Print predicator
    //         printf("predicator block(%d, %d) thread(%d) thread_block_idx_x %d thread_block_idx_y %d, %d %d %s\n", blockIdx.x, blockIdx.y, threadIdx.x, thread_block_idx_x, thread_block_idx_y, i, j, register_fragment_predicator(i, j) ? "true" : "false");
    //     }
    // }
    
    // for (unsigned int i{0}; i < cute::size<0>(thread_tile_src); ++i)
    // {
    //     for (unsigned int j{0}; j < cute::size<1>(thread_tile_src); ++j)
    //     {
    //         register_fragment_predicator(i, j) = cute::get<0>(thread_tile_src(i, j)) < 1021 &&
    //                                             cute::get<1>(thread_tile_src(i, j)) < 2049;
    //     }
    // }

    // auto predicate{cute::make_tensor_like<bool>(thread_tile_src)};

    // for (unsigned int i{0}; i < cute::size<0>(thread_tile_src); ++i)
    // {
    //     for (unsigned int j{0}; j < cute::size<1>(thread_tile_src); ++j)
    //     {
    //         predicate(i, j) = cute::get<0>(thread_tile_src(i, j)) < 1021 &&
    //                           cute::get<1>(thread_tile_src(i, j)) < 2049;
    //     }
    // }

    // cute::copy(thread_tile_src, register_fragment);
    // cute::copy(register_fragment, thread_tile_dst_transposed);

    // cute::copy_if(register_fragment_predicator, thread_tile_src, register_fragment);
    // cute::copy_if(register_fragment_predicator, register_fragment, thread_tile_dst_transposed);

    // cute::copy_if(register_fragment_predicator, thread_tile_src, thread_tile_dst_transposed);
    printf("Copying----------------------------\n");
    // for (unsigned int i{0}; i < cute::size<0>(register_fragment_predicator); ++i)
    // {
    //     for (unsigned int j{0}; j < cute::size<1>(register_fragment_predicator); ++j)
    //     {
    //         // printf("Copying i %d j %d predicator %s input value %d \n", i, j, register_fragment_predicator(i, j) ? "true" : "false", thread_tile_src(i, j));
    //         if (register_fragment_predicator(i, j))
    //         {
    //             printf("predicator block(%d, %d) thread(%d) thread_block_idx_x %d thread_block_idx_y %d, Copying i %d j %d predicator %s input value %d \n", blockIdx.x, blockIdx.y, threadIdx.x, thread_block_idx_x, thread_block_idx_y, i, j, register_fragment_predicator(i, j) ? "true" : "false", thread_tile_src(i, j));
    //             thread_tile_dst_transposed(i, j) = thread_tile_src(i, j);
    //         }
    //     }
    // }

    for (unsigned int i{0}; i < cute::size<0>(predicator); ++i)
    {
        for (unsigned int j{0}; j < cute::size<1>(predicator); ++j)
        {
            // printf("Copying i %d j %d predicator %s input value %d \n", i, j, register_fragment_predicator(i, j) ? "true" : "false", thread_tile_src(i, j));
            if (predicator(i, j))
            {
                printf("predicator block(%d, %d) thread(%d) thread_block_idx_x %d thread_block_idx_y %d, Copying i %d j %d predicator %s input value %d \n", blockIdx.x, blockIdx.y, threadIdx.x, thread_block_idx_x, thread_block_idx_y, i, j, predicator(i, j) ? "true" : "false", thread_tile_src(i, j));
                thread_tile_dst_transposed(i, j) = thread_tile_src(i, j);
            }
        }
    }

    



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
        if (data[i] != ref[i])
        {
            std::cout << i << " " << data[i] << " " << ref[i] << std::endl;
            return false;
        }
    }

    return true;
}

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

    // unsigned int const M{2048}; // Number of columns.
    // unsigned int const N{1024}; // Number of rows.

    // unsigned int const M{2049}; // Number of columns.
    // unsigned int const N{1021}; // Number of rows.

    // unsigned int const M{16}; // Number of columns.
    // unsigned int const N{8}; // Number of rows.

    unsigned int const M{74}; // Number of columns.
    unsigned int const N{13}; // Number of rows.

    // unsigned int const M{10}; // Number of columns.
    // unsigned int const N{7}; // Number of rows.

    // unsigned int const M{16}; // Number of columns.
    // unsigned int const N{8}; // Number of rows.

    auto const tensor_shape{cute::make_shape(N, M)};
    auto const tensor_shape_transposed{cute::make_shape(M, N)};

    thrust::host_vector<Element> h_src(cute::size(tensor_shape));
    thrust::host_vector<Element> h_dst(cute::size(tensor_shape_transposed));
    thrust::host_vector<Element> h_dst_ref(cute::size(tensor_shape_transposed));

    initialize(h_src.data(), h_src.size());
    transpose(h_src.data(), h_dst_ref.data(), M, N);

    thrust::device_vector<Element> d_src{h_src};
    thrust::device_vector<Element> d_dst{h_dst};

    auto const global_memory_layout_src{cute::make_layout(
        tensor_shape, cute::GenRowMajor{})}; // (N, M) : (M, 1)
    auto const global_memory_layout_dst{cute::make_layout(
        tensor_shape_transposed, cute::GenRowMajor{})}; // (M, N) : (N, 1)
    auto const global_memory_layout_dst_transposed{cute::make_layout(
        tensor_shape, cute::GenColMajor{})}; // (N, M) : (1, N)

    cute::print(global_memory_layout_src);
    std::cout << std::endl;
    cute::print(global_memory_layout_dst);
    std::cout << std::endl;
    cute::print(global_memory_layout_dst_transposed);
    std::cout << std::endl;

    auto const tensor_src{cute::make_tensor(
        cute::make_gmem_ptr(thrust::raw_pointer_cast(d_src.data())),
        global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(
        cute::make_gmem_ptr(thrust::raw_pointer_cast(d_dst.data())),
        global_memory_layout_dst)};
    auto const tensor_dst_transposed{cute::make_tensor(
        cute::make_gmem_ptr(thrust::raw_pointer_cast(d_dst.data())),
        global_memory_layout_dst_transposed)};

    using TILE_SIZE_X = cute::Int<8>;
    using TILE_SIZE_Y = cute::Int<4>;

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_Y{}, TILE_SIZE_X{})};
    constexpr auto block_shape_transposed{
        cute::make_shape(TILE_SIZE_X{}, TILE_SIZE_Y{})};

    auto const tiled_tensor_src{cute::tiled_divide(
        tensor_src, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), N /
                                   // TILE_SIZE_Y, M / TILE_SIZE_X)
    auto const tiled_tensor_dst_transposed{cute::tiled_divide(
        tensor_dst_transposed, block_shape)}; // ((TILE_SIZE_Y, TILE_SIZE_X), N
                                              // / TILE_SIZE_Y, M / TILE_SIZE_X)
    std::cout << "== Tiled tensor ==" << std::endl;
    cute::print(tiled_tensor_src);
    std::cout << std::endl;
    cute::print(tiled_tensor_dst_transposed);
    std::cout << std::endl;

    // auto const g_src_example{
    //     tiled_tensor_src(cute::make_coord(cute::_, cute::_), 0, 0)};
    // auto const g_dst_example{
    //     tiled_tensor_dst_transposed(cute::make_coord(cute::_, cute::_), 0, 0)};
    auto const g_src_example{
        tiled_tensor_src(cute::make_coord(cute::_, cute::_), 1, 2)};
    auto const g_dst_example{
        tiled_tensor_dst_transposed(cute::make_coord(cute::_, cute::_), 1, 2)};
    std::cout << "ff----------------" << std::endl;
    cute::print(g_src_example);
    std::cout << std::endl;
    cute::print(g_dst_example);
    std::cout << std::endl;
    // std::cout << "Make identity tensor" << std::endl;
    // auto const make_identity_tensor(g_src_example);
    constexpr auto global_tile_columns{cute::size<1>(g_src_example)};
    constexpr auto global_tile_rows{cute::size<0>(g_src_example)};
    std::cout << "global_tile_columns: " << global_tile_columns << std::endl;
    std::cout << "global_tile_rows: " << global_tile_rows << std::endl;
    printf("global_tile_columns: %d\n", global_tile_columns);
    printf("global_tile_rows: %d\n", global_tile_rows);

    using THREAD_BLOCK_SIZE_X = cute::Int<4>;
    using THREAD_BLOCK_SIZE_Y = cute::Int<2>;

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    CUTE_STATIC_ASSERT(TILE_SIZE_Y::value % THREAD_BLOCK_SIZE_Y::value == 0,
                       "TILE_SIZE_Y must be divisible by THREAD_BLOCK_SIZE_Y");

    constexpr auto thread_block_shape{
        cute::make_shape(THREAD_BLOCK_SIZE_Y{}, THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_layout{
        cute::make_layout(thread_block_shape, cute::GenRowMajor{})};

    auto const thread_tile_src_example{
        cute::local_partition(g_src_example, thread_layout, 0)};
    auto const thread_tile_dst_transposed_example{
        cute::local_partition(g_dst_example, thread_layout, 0)};
    cute::print(thread_tile_src_example);
    std::cout << std::endl;
    cute::print(thread_tile_dst_transposed_example);
    std::cout << std::endl;

    auto const fragment_src_example{
        cute::make_tensor_like(thread_tile_src_example)};
    cute::print(fragment_src_example);
    std::cout << std::endl;

    auto fragment_src_example_identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(fragment_src_example),
                         cute::size<1>(fragment_src_example)))};
    cute::print(fragment_src_example_identity_tensor);
    
    for (unsigned int i{0}; i < cute::size<0>(fragment_src_example_identity_tensor); ++i)
    {
        for (unsigned int j{0}; j < cute::size<1>(fragment_src_example_identity_tensor); ++j)
        {
            std::cout << i << " " << j << " " << cute::get<0>(fragment_src_example_identity_tensor(i, j)) << " " << cute::get<1>(fragment_src_example_identity_tensor(i, j)) << std::endl;
        }
    }

    std::cout << "----------------" << std::endl;

    for (unsigned int i{0}; i < cute::size(fragment_src_example_identity_tensor); ++i)
    {
        std::cout << i << " " << fragment_src_example_identity_tensor(i) << std::endl;
    }


    // auto register_fragment_predicator{cute::make_tensor<bool>(
    //     cute::make_shape(cute::size<0>(fragment_src_example),
    //                      cute::size<1>(fragment_src_example)))};
    
    // for (unsigned int i{0}; i < cute::size<0>(fragment_src_example_identity_tensor); ++i)
    // {
    //     for (unsigned int j{0}; j < cute::size<1>(fragment_src_example_identity_tensor); ++j)
    //     {
    //         register_fragment_predicator(i, j) = cute::get<0>(fragment_src_example_identity_tensor(i, j)) < num_max_rows - blockIdx.y * global_tile_rows - thread_block_idx_y * thread_tile_rows &&
    //                                             cute::get<1>(fragment_src_example_identity_tensor(i, j)) < num_max_columns - blockIdx.x * global_tile_columns - thread_block_idx_x * thread_tile_columns;
    //     }
    // }



    dim3 const grid_dim{cute::size<2>(tiled_tensor_src),
                        cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    auto const num_max_columns{cute::stride<0>(g_src_example)};
    auto const num_max_rows{cute::stride<1>(g_dst_example)};

    // std::cout << "num_max_columns: " << num_max_columns << std::endl;
    // std::cout << "num_max_rows: " << num_max_rows << std::endl;

    transpose_naive<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst_transposed, thread_layout);
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

    // print(h_dst.data(), h_dst_ref.data(), h_dst.size());
}