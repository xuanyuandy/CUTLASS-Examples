#ifndef CUTE_VECTOR_COPY_HPP
#define CUTE_VECTOR_COPY_HPP

#include <cuda_runtime.h>

template <typename T>
cudaError_t launch_vector_copy(T const* input_vector, T* output_vector,
                               unsigned int size, cudaStream_t stream);

template <typename T>
cudaError_t launch_vector_copy_vectorized(T const* input_vector,
                                          T* output_vector, unsigned int size,
                                          cudaStream_t stream);

#endif // CUTE_VECTOR_COPY_HPP