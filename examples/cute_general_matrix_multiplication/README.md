# CuTe General Matrix Multiplication

## Introduction

These examples demonstrate the implementation of general matrix transpose kernels using the CuTe. They follow the cuBLAS interface design and supports matrices that are transposed or not transposed. The naive kernels perform boundary checks and can be used for any matrix dimension size. The kernels that performs tiled copy assumes each matrix dimension size is a multiple of certain size, usually 32, depending on the data type.

The pipelining optimization for general matrix multiplication remains to be implemented.

The performance was mainly optimized for the NT case where the matrix A is a M x K column-major matrix, the matrix B is a K x N row-major matrix, and the matrix C is a M x N column-major matrix. I will try coming up with a general implementation that optimizes for all the NN, NT, TN, and TT cases in the future. An idea is to use shared memory layouts that are compatible with the global memory layouts so that global memory vectorized access can always be enabled.

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllGeneralMatrixMultiplication.*" --verbose
The following tests passed:
        TestAllGeneralMatrixMultiplicationNaive
        TestAllGeneralMatrixMultiplicationNaiveTiledCopyTiledMma
        TestAllGeneralMatrixMultiplicationTensorCoreTiledCopyTiledMma

100% tests passed, 0 tests failed out of 3
```

### Run Performance Measurement

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllGeneralMatrixMultiplication.*" --verbose
```

Only the general matrix multiplication kernels that consume half typed data and compute in half precision are documented here for a problem size of 4096 x 4096 x 4096 on NVIDIA GeForce RTX 3090.

|                Kernel Name                | Trans A | Trans B | Latency (ms) |  TOPs   | Performance VS cuBLAS (%) |
| :---------------------------------------: | :-----: | :-----: | :----------: | :-----: | :-----------------------: |
|                  cuBLAS                   |    T    |    T    |   0.961536   | 142.937 |             -             |
|                  cuBLAS                   |    T    |    N    |   1.00198    | 137.167 |             -             |
|                  cuBLAS                   |    N    |    T    |   0.942925   | 145.758 |             -             |
|                  cuBLAS                   |    N    |    N    |   1.06793    | 128.697 |             -             |
|                   Naive                   |    T    |    T    |   13.3946    | 10.2607 |          7.17852          |
|                   Naive                   |    T    |    N    |   12.9244    | 10.6341 |          7.75264          |
|                   Naive                   |    N    |    T    |   13.9895    | 9.82445 |          6.74024          |
|                   Naive                   |    N    |    N    |   9.64628    | 14.2479 |          11.0709          |
|          Tiled Copy + Tiled MMA           |    T    |    T    |   7.89575    | 17.4067 |          12.2555          |
|          Tiled Copy + Tiled MMA           |    T    |    N    |   9.76526    | 14.0743 |          10.375           |
|          Tiled Copy + Tiled MMA           |    N    |    T    |   5.58715    | 24.5991 |          17.4957          |
|          Tiled Copy + Tiled MMA           |    N    |    N    |   9.00575    | 15.2612 |          11.0451          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    T    |    T    |   5.25312    | 26.1633 |          20.4351          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    T    |    N    |   8.35786    | 16.4443 |          13.5997          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    N    |    T    |   2.66874    | 51.4996 |          40.6004          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    N    |    N    |   5.03395    | 27.3024 |          22.1927          |

### Run Nsight Compute Profiling

```bash
for file in build/examples/cute_general_matrix_multiplication/tests/profile_*; do
    filename=$(basename -- "$file")
    ncu --set full -f -o ncu_reports/"$filename" "$file"
done
```

### Run Compute Sanitizer

```bash
for file in build/examples/cute_general_matrix_multiplication/tests/test_*; do
    compute-sanitizer --leak-check full "$file"
done
```

## References
