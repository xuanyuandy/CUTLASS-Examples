#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/array.h>

#include "cute_vector_copy.hpp"

template <class TENSOR_SRC, class TENSOR_DST, class THREAD_LAYOUT,
          class VECTOR_LAYOUT>
static __global__ void vector_copy_vectorized(TENSOR_SRC tensor_src,
                                              TENSOR_DST tensor_dst,
                                              THREAD_LAYOUT, VECTOR_LAYOUT)
{
    using Element = typename TENSOR_SRC::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_), blockIdx.x)};
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_), blockIdx.x)};

    auto thread_global_tile_src{
        cute::local_partition(global_tile_src, THREAD_LAYOUT{}, threadIdx.x)};
    auto thread_global_tile_dst{
        cute::local_partition(global_tile_dst, THREAD_LAYOUT{}, threadIdx.x)};

    using AccessType =
        cutlass::AlignedArray<Element, cute::size(VECTOR_LAYOUT{})>;
    using CopyAtom = cute::Copy_Atom<cute::UniversalCopy<AccessType>, Element>;
    auto tiled_copy{
        cute::make_tiled_copy(CopyAtom{}, THREAD_LAYOUT{}, VECTOR_LAYOUT{})};
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x);

    auto thread_tile_src = thread_copy.partition_S(global_tile_src);
    auto thread_tile_dst = thread_copy.partition_D(global_tile_dst);

    auto fragment{cute::make_fragment_like(thread_tile_dst)};

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("global_tile_src\n");
        cute::print(global_tile_src);
        printf("\n");
        printf("global_tile_dst\n");
        cute::print(global_tile_dst);
        printf("\n");
        printf("tiled_copy\n");
        cute::print(tiled_copy);
        printf("\n");
        printf("thread_copy\n");
        cute::print(thread_copy);
        printf("\n");

        printf("thread_tile_src\n");
        cute::print(thread_tile_src);
        printf("\n");
        printf("thread_tile_dst\n");
        cute::print(thread_tile_dst);
        printf("\n");
    }

    cute::copy(tiled_copy, thread_tile_src, fragment);
    cute::copy(tiled_copy, fragment, thread_tile_dst);
    // cute::copy(tiled_copy, thread_tile_src, thread_tile_dst);
}

template <typename T, typename VEC_TYPE>
static cudaError_t
launch_vector_copy_vectorized(T const* input_vector, T* output_vector,
                              unsigned int size, cudaStream_t stream)
{
    static_assert(sizeof(VEC_TYPE) % sizeof(T) == 0,
                  "sizeof(VEC_TYPE) must be a multiple of sizeof(T)");
    constexpr unsigned int NUM_VECTOR_ELEMENTS{sizeof(VEC_TYPE) / sizeof(T)};

    auto const tensor_shape{cute::make_shape(size)};
    auto const global_memory_layout_src{cute::make_layout(tensor_shape)};
    auto const global_memory_layout_dst{cute::make_layout(tensor_shape)};

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_vector),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_vector),
                                            global_memory_layout_dst)};
    cute::print(tensor_src);
    printf("\n");

    using TILE_SIZE_X = cute::Int<2048>;

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_X{})};

    auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)};
    auto const tiled_tensor_dst{cute::tiled_divide(tensor_dst, block_shape)};

    cute::print(tiled_tensor_src);
    printf("\n");

    using THREAD_BLOCK_SIZE_X = cute::Int<256>;

    constexpr auto thread_block_shape{cute::make_shape(THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_layout{cute::make_layout(thread_block_shape)};

    using VECTOR_SIZE_X = cute::Int<NUM_VECTOR_ELEMENTS>;
    using VECTOR_SIZE_Y = cute::Int<1>;
    constexpr auto vector_shape{cute::make_shape(VECTOR_SIZE_X{})};
    constexpr auto vector_layout{cute::make_layout(vector_shape)};

    constexpr auto vector_shape_1{
        cute::make_shape(cute::Int<4>{}, cute::Int<1>{})};
    constexpr auto vector_layout_1{cute::make_layout(vector_shape_1)};

    constexpr auto thread_block_shape_1{
        cute::make_shape(cute::Int<32>{}, cute::Int<8>{})};
    constexpr auto thread_layout_1{cute::make_layout(thread_block_shape_1)};

    // using AccessType = cutlass::AlignedArray<float,
    // cute::size(thread_layout_1)>; using CopyAtom =
    // cute::Copy_Atom<cute::UniversalCopy<AccessType>, float>; auto
    // tiled_copy{cute::make_tiled_copy(CopyAtom{}, thread_layout_1,
    // vector_layout_1)};

    // Define `AccessType` which controls the size of the actual memory access.
    using AccessType =
        cutlass::AlignedArray<float, cute::size(vector_layout_1)>;

    // A copy atom corresponds to one hardware memory access.
    using Atom = cute::Copy_Atom<cute::UniversalCopy<AccessType>, float>;

    // Construct tiled copy, a tiling of copy atoms.
    //
    // Note, this assumes the vector and thread layouts are aligned with
    // contigous data in GMEM. Alternative thread layouts are possible but may
    // result in uncoalesced reads. Alternative vector layouts are also
    // possible, though incompatible layouts will result in compile time errors.
    auto tiled_copy =
        cute::make_tiled_copy(Atom{},           // access size
                              thread_layout_1,  // thread layout
                              vector_layout_1); // vector layout (e.g. 4x1)

    dim3 const grid_dim{cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    vector_copy_vectorized<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst, thread_layout, vector_layout);

    return cudaGetLastError();
}

// Explicit instantiation.
template cudaError_t launch_vector_copy_vectorized<float, int4>(
    float const* input_vector, float* output_vector, unsigned int size,
    cudaStream_t stream);
// template cudaError_t launch_vector_copy_vectorized<double, int4>(double
// const* input_vector,
//                                                 double* output_vector,
//                                                 unsigned int size,
//                                                 cudaStream_t stream);