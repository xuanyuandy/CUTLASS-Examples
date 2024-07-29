# CUTLASS Docker

Docker Image for CUTLASS Applications

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

### Build CUDA Kernels

To build the CUDA kernels, please run the following commands inside the Docker container.

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
```
