//#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>

// Typedefs for the original functions
typedef CUresult (*orig_cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, CUstream, void **, void **);
typedef cudaError_t (*orig_cudaFuncGetAttributes_t)(struct cudaFuncAttributes *, const void *);
typedef cudaError_t (*orig_cudaMemcpyAsync_t)(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);



void printInterceptInfo(const char *functionName, CUstream stream, const char *str) {
    printf("Intercepted %s: stream=%p, info=%s\n", functionName, (void *)stream, str);
}

// Intercept cuLaunchKernel
/*CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    static orig_cuLaunchKernel_t orig_cuLaunchKernel = NULL;
    if (!orig_cuLaunchKernel) {
        orig_cuLaunchKernel = (orig_cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    }

//    printf("Intercepted cuLaunchKernel: gridDim=(%u, %u, %u), blockDim=(%u, %u, %u)\n",
//           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);

    return orig_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                               sharedMemBytes, hStream, kernelParams, extra);
}*/
*/CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    static orig_cuLaunchKernel_t orig_cuLaunchKernel = NULL;
    if (!orig_cuLaunchKernel) {
        orig_cuLaunchKernel = (orig_cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
        if (!orig_cuLaunchKernel) {
            fprintf(stderr, "Error in dlsym: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }

    if (!f || (!kernelParams && !extra)) {
        fprintf(stderr, "Invalid parameters passed to cuLaunchKernel\n");
        exit(EXIT_FAILURE);
    }

    char infoStr[128];
    char *p = infoStr;
    p += sprintf(p, "gridDim=(%u, %u, %u), ", gridDimX, gridDimY, gridDimZ);
    sprintf(p, "blockDim=(%u, %u, %u)", blockDimX, blockDimY, blockDimZ);
    printInterceptInfo("cuLaunchKernel", hStream, infoStr);

    return orig_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                               sharedMemBytes, hStream, kernelParams, extra);
}

// Intercept cudaFuncGetAttributes
/*cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
    static orig_cudaFuncGetAttributes_t orig_cudaFuncGetAttributes = NULL;
    if (!orig_cudaFuncGetAttributes) {
        orig_cudaFuncGetAttributes = (orig_cudaFuncGetAttributes_t)dlsym(RTLD_NEXT, "cudaFuncGetAttributes");
    }

    printf("Intercepted cudaFuncGetAttributes\n");
    return orig_cudaFuncGetAttributes(attr, func);
}

// Intercept cudaMemcpyAsync
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    static orig_cudaMemcpyAsync_t orig_cudaMemcpyAsync = NULL;
    if (!orig_cudaMemcpyAsync) {
        orig_cudaMemcpyAsync = (orig_cudaMemcpyAsync_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    }

    printf("Intercepted cudaMemcpyAsync: count=%zu\n", count);
    return orig_cudaMemcpyAsync(dst, src, count, kind, stream);
}

// Compile with:
// gcc -shared -o libcudaintercept.so -fPIC cudaintercept.c -ldl
// Usage:
// LD_PRELOAD=./libcudaintercept.so <your_cuda_application>
*/
