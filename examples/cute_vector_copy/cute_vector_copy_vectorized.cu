#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_vector_copy.hpp"

template <class TENSOR_SRC, class TENSOR_DST, class TILED_COPY>
static __global__ void
vector_copy_vectorized(TENSOR_SRC tensor_src, TENSOR_DST tensor_dst,
                       unsigned int size, TILED_COPY tiled_copy)
{
    using Element = typename TENSOR_SRC::value_type;

    auto global_tile_src{tensor_src(cute::make_coord(cute::_), blockIdx.x)};
    auto global_tile_dst{tensor_dst(cute::make_coord(cute::_), blockIdx.x)};

    auto thread_copy{tiled_copy.get_thread_slice(threadIdx.x)};

    auto thread_tile_src{thread_copy.partition_S(
        global_tile_src)}; // (CopyAtomShape, NumCopyTile)
    auto thread_tile_dst{thread_copy.partition_D(
        global_tile_dst)}; // (CopyAtomShape, NumCopyTile)

    auto const identity_tensor{cute::make_identity_tensor(
        cute::make_shape(cute::size(global_tile_src)))};
    auto thread_identity_tensor_src{thread_copy.partition_S(
        identity_tensor)}; // (CopyAtomShape, NumCopyTile)
    auto thread_identity_tensor_dst{thread_copy.partition_D(
        identity_tensor)}; // (CopyAtomShape, NumCopyTile)

    auto fragment{cute::make_fragment_like(
        thread_tile_src)}; // (CopyAtomShape, NumCopyTile)
    auto predicator{cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(thread_tile_src)))}; // (NumCopyTile)

    for (unsigned int i{0}; i < cute::size(predicator); ++i)
    {
        auto const thread_identity{thread_identity_tensor_src(0, i)};
        bool const is_in_bound{cute::get<0>(thread_identity) +
                                   blockIdx.x * cute::size(global_tile_src) <
                               size};
        predicator(i) = is_in_bound;
    }

    cute::copy_if(tiled_copy, predicator, thread_tile_src, fragment);
    cute::copy_if(tiled_copy, predicator, fragment, thread_tile_dst);
    // Alternatively, we could just do the following instead.
    // cute::copy_if(tiled_copy, predicator, thread_tile_src, thread_tile_dst);
}

template <typename T>
static cudaError_t
launch_vector_copy_vectorized(T const* input_vector, T* output_vector,
                              unsigned int size, cudaStream_t stream)
{
    using VECTOR_TYPE = cute::uint128_t;
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0,
                  "sizeof(VECTOR_TYPE) must be a multiple of sizeof(T)");
    constexpr unsigned int NUM_VECTOR_ELEMENTS{sizeof(VECTOR_TYPE) / sizeof(T)};

    if (size % NUM_VECTOR_ELEMENTS != 0)
    {
        return cudaErrorInvalidValue;
    }

    auto const tensor_shape{cute::make_shape(size)};
    auto const global_memory_layout_src{cute::make_layout(tensor_shape)};
    auto const global_memory_layout_dst{cute::make_layout(tensor_shape)};

    auto const tensor_src{cute::make_tensor(cute::make_gmem_ptr(input_vector),
                                            global_memory_layout_src)};
    auto const tensor_dst{cute::make_tensor(cute::make_gmem_ptr(output_vector),
                                            global_memory_layout_dst)};

    using TILE_SIZE_X = cute::Int<2048>;

    constexpr auto block_shape{cute::make_shape(TILE_SIZE_X{})};

    auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)};
    auto const tiled_tensor_dst{cute::tiled_divide(tensor_dst, block_shape)};

    using THREAD_BLOCK_SIZE_X = cute::Int<256>;

    constexpr auto thread_block_shape{cute::make_shape(THREAD_BLOCK_SIZE_X{})};
    constexpr auto thread_layout{cute::make_layout(thread_block_shape)};

    using VECTOR_SIZE_X = cute::Int<NUM_VECTOR_ELEMENTS>;
    constexpr auto vector_shape{cute::make_shape(VECTOR_SIZE_X{})};
    // Copy atom vector layout.
    constexpr auto vector_layout{cute::make_layout(vector_shape)};

    // Construct tiled copy, a tiling of copy atoms.
    using AccessType = cutlass::AlignedArray<T, cute::size(vector_layout)>;
    // A copy atom corresponds to one hardware memory access.
    using CopyAtom = cute::Copy_Atom<cute::UniversalCopy<AccessType>, T>;
    // Construct tiled copy, a tiling of copy atoms.
    auto tiled_copy{
        cute::make_tiled_copy(CopyAtom{}, thread_layout, vector_layout)};

    dim3 const grid_dim{cute::size<1>(tiled_tensor_src)};
    dim3 const thread_dim{cute::size(thread_layout)};

    CUTE_STATIC_ASSERT(TILE_SIZE_X::value % THREAD_BLOCK_SIZE_X::value == 0,
                       "TILE_SIZE_X must be divisible by THREAD_BLOCK_SIZE_X");
    vector_copy_vectorized<<<grid_dim, thread_dim, 0, stream>>>(
        tiled_tensor_src, tiled_tensor_dst, size, tiled_copy);

    return cudaGetLastError();
}

// Explicit instantiation.
template cudaError_t
launch_vector_copy_vectorized<float>(float const* input_vector,
                                     float* output_vector, unsigned int size,
                                     cudaStream_t stream);
template cudaError_t
launch_vector_copy_vectorized<double>(double const* input_vector,
                                      double* output_vector, unsigned int size,
                                      cudaStream_t stream);