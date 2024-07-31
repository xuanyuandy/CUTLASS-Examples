# CUTLASS Examples

## Build Examples

To build the CUDA kernels, please run the following commands inside the Docker container.

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
```

## Run Examples

To run the FP32 and FP16 GEMM CUDA kernels, please run the following commands inside the Docker container.

```bash
$ ./build/gemm_api_v2/CUTLASS-GEMM-API-V2
$ ./build/gemm_api_v3/CUTLASS-GEMM-API-V3
```
