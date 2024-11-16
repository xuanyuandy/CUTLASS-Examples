# CuTe Matrix Transpose

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllMatrixTranspose.*" --verbose
```

### Run Performance Measurement

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllMatrixTranspose.*" --verbose
```

### Run Nsight Compute Profiling

```bash
$ ncu --set full -f -o ncu_reports/matrix_transpose build/examples/cute_transpose/tests/profile_matrix_transpose_ncu
```
