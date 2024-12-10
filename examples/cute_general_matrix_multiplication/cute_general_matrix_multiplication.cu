#include <iostream>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "cute_general_matrix_multiplication.hpp"

// template <class TA, class TB, class TC, class Alpha, class Beta>
// void launch_gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA,
//                     TB const* B, int ldB, Beta beta, TC* C, int ldC,
//                     cudaStream_t stream)
// {
// }