#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Convenience macro for launching CUDA kernels with stream
#define ATARAXOS_CUDA_LAUNCH(kernel, grid, block, stream, ...) \
    do { \
        kernel<<<grid, block, 0, stream>>>(__VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } while (0)
