#ifndef CUTE_TRANSPOSE_HPP
#define CUTE_TRANSPOSE_HPP

#include <cuda_runtime.h>

template <typename T>
cudaError_t launch_transpose_naive(T const* input_matrix, T* output_matrix,
                                   unsigned int M, unsigned int N,
                                   cudaStream_t stream);

template <typename T>
cudaError_t launch_transpose_shared_memory_bank_conflicts(T const* input_matrix,
                                                          T* output_matrix,
                                                          unsigned int M,
                                                          unsigned int N,
                                                          cudaStream_t stream);

#endif // CUTE_TRANSPOSE_HPP