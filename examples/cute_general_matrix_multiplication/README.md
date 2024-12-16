# CuTe General Matrix Multiplication

## Introduction

These examples demonstrate the implementation of general matrix transpose kernels using the CuTe. They follow the cuBLAS interface design and supports matrices that are transposed or not transposed. The naive kernels perform boundary checks and can be used for any matrix dimension size. The kernels that performs tiled copy assumes each matrix dimension size is a multiple of certain size, usually 32, depending on the data type.

The pipelining optimization for general matrix multiplication remains to be implemented.

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
|                  cuBLAS                   |    T    |    T    |   0.965811   | 142.304 |             -             |
|                  cuBLAS                   |    T    |    N    |   1.16111    | 118.368 |             -             |
|                  cuBLAS                   |    N    |    T    |   0.953651   | 144.119 |             -             |
|                  cuBLAS                   |    N    |    N    |   1.03472    | 132.827 |             -             |
|                   Naive                   |    T    |    T    |   14.5984    | 9.41468 |          6.61589          |
|                   Naive                   |    T    |    N    |   11.8134    | 11.6342 |          9.8288           |
|                   Naive                   |    N    |    T    |    13.864    | 9.91334 |          6.8786           |
|                   Naive                   |    N    |    N    |   10.1447    | 13.5478 |          10.1996          |
|          Tiled Copy + Tiled MMA           |    T    |    T    |   9.22368    | 14.9007 |          13.0754          |
|          Tiled Copy + Tiled MMA           |    T    |    N    |   10.5959    | 12.9709 |          9.51138          |
|          Tiled Copy + Tiled MMA           |    N    |    T    |   5.51072    | 24.9403 |          17.5581          |
|          Tiled Copy + Tiled MMA           |    N    |    N    |   8.28507    | 16.5888 |          11.9505          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    T    |    T    |   5.95068    | 23.0963 |          17.0876          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    T    |    N    |   8.30013    | 16.5586 |          12.0101          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    N    |    T    |   3.30701    | 41.5599 |          32.5402          |
| Tiled Copy + Tiled MMA + SM80 Tensor Core |    N    |    N    |   5.85392    | 23.4781 |          19.3503          |

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
