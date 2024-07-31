# CUTLASS Docker

## Introduction

The Docker and CMake examples for [CUTLASS](https://github.com/NVIDIA/cutlass) library.

## CUTLASS Docker Container

Docker is used to build and run the CUDA kernels. The custom Docker container is built based on the [NVIDIA NGC CUDA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) 12.2.2 Docker container.

Please adjust the base Docker container CUDA version if the host computer has a different CUDA version. Otherwise, weird compilation errors and runtime errors may occur.

### Set CUTLASS Version

To set the CUTLASS version, please run the following command.

```bash
$ CUTLASS_VERSION=3.5.0
```

### Build Docker Images

To build the custom Docker image, please run the following command.

```bash
$ docker build -f docker/cutlass.Dockerfile --no-cache --build-arg CUTLASS_VERSION=${CUTLASS_VERSION} --tag=cutlass:${CUTLASS_VERSION} .
```

### Run Docker Container

To run the custom Docker container, please run the following command.

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt cutlass:${CUTLASS_VERSION}
```

## CUTLASS CMake Examples

Inside the CUTLASS Docker container, follow the [README](/examples/README.md) in the [examples](/examples/).

## References

- [CUTLASS Docker Container](https://leimao.github.io/blog/CUTLASS-Docker/)
