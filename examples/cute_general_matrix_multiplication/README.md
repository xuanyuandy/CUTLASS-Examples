# CuTe General Matrix Multiplication

## Introduction

These examples demonstrate the implementation of general matrix transpose kernels using the CuTe. Each kernel does boundary checks and can be used for any matrix size. The matrices are assumed to be stored in row-major order. The general matrix transpose vectorized kernel also assumes the row size is a multiple of certain size depending on the data type.

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllGeneralMatrixMultiplication.*" --verbose
```

### Run Performance Measurement

### Run Nsight Compute Profiling

### Run Compute Sanitizer

## References
