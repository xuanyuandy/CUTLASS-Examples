#ifndef CUTE_GENERAL_MATRIX_MULTIPLICATION_HPP
#define CUTE_GENERAL_MATRIX_MULTIPLICATION_HPP

#include <cuda_runtime.h>

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_gemm_naive(char transA, char transB, int m, int n, int k,
                              Alpha alpha, TA const* A, int ldA, TB const* B,
                              int ldB, Beta beta, TC* C, int ldC,
                              cudaStream_t stream);

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_gemm_naive_tiled_copy(char transA, char transB, int m, int n,
                                         int k, Alpha alpha, TA const* A,
                                         int ldA, TB const* B, int ldB,
                                         Beta beta, TC* C, int ldC,
                                         cudaStream_t stream);

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t
launch_gemm_naive_tiled_copy_tiled_mma(char transA, char transB, int m, int n,
                                       int k, Alpha alpha, TA const* A, int ldA,
                                       TB const* B, int ldB, Beta beta, TC* C,
                                       int ldC, cudaStream_t stream);

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_sgemm_1(char transA, char transB, int m, int n, int k,
                           Alpha alpha, TA const* A, int ldA, TB const* B,
                           int ldB, Beta beta, TC* C, int ldC,
                           cudaStream_t stream);

template <class TA, class TB, class TC, class Alpha, class Beta>
cudaError_t launch_sgemm_2(char transA, char transB, int m, int n, int k,
                           Alpha alpha, TA const* A, int ldA, TB const* B,
                           int ldB, Beta beta, TC* C, int ldC,
                           cudaStream_t stream);

#endif // CUT_GENERAL_MATRIX_MULTIPLICATION_HPP