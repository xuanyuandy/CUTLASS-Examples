# CuTe Matrix Transpose

## Introduction

These examples demonstrate the implementation of general matrix transpose kernels using the CuTe. Each kernel does boundary checks and can be used for any matrix size.

## Usages

### Run Unit Tests

```bash
$ ctest --test-dir build/ --tests-regex "TestAllMatrixTranspose.*" --verbose

The following tests passed:
        TestAllMatrixTransposeNaiveCoalescedRead
        TestAllMatrixTransposeNaiveCoalescedWrite
        TestAllMatrixTransposeSharedMemoryBankConflictRead
        TestAllMatrixTransposeSharedMemoryBankConflictWrite
        TestAllMatrixTransposeSharedMemoryPadded
        TestAllMatrixTransposeSharedMemorySwizzled

100% tests passed, 0 tests failed out of 6
```

### Run Performance Measurement

```bash
$ ctest --test-dir build/ --tests-regex "ProfileAllMatrixTranspose.*" --verbose
```

The following tables show the performance measurements of the matrix transpose kernels on NVIDIA GeForce RTX 3090.

|            Kernel Name            | Latency (ms) | Effective Bandwidth (GB/s) | Peak Bandwidth Percentage (%) |
| :-------------------------------: | :----------: | :------------------------: | :---------------------------: |
|       Naive Coalesced Read        |   7.75992    |          257.734           |            27.5329            |
|       Naive Coalesced Write       |   3.12904    |          639.174           |            68.2809            |
| Shared Memory Bank Conflict Read  |   2.98797    |          669.351           |            71.5045            |
| Shared Memory Bank Conflict Write |    2.9763    |          671.976           |            71.7849            |
|       Shared Memory Padded        |   2.98273    |          670.527           |            71.6302            |
|      Shared Memory Swizzled       |   2.92828    |          682.994           |            72.962             |

### Run Nsight Compute Profiling

```bash
for file in build/examples/cute_transpose/tests/profile_*; do
    filename=$(basename -- "$file")
    ncu --set full -f -o ncu_reports/"$filename" "$file"
done
```
