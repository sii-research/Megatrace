#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

/**
 * GPU Vendor Configuration
 * 
 * Define GPU vendor at compile time:
 * - For NVIDIA:   g++ -DMEGATRACE_GPU_NVIDIA ...
 * - For AMD:      g++ -DMEGATRACE_GPU_AMD ...
 * 
 * If no vendor is specified, defaults to NVIDIA for backward compatibility.
 */

// Default to NVIDIA if no vendor is specified
#if !defined(MEGATRACE_GPU_NVIDIA) && !defined(MEGATRACE_GPU_AMD)
    #define MEGATRACE_GPU_NVIDIA
#endif

// Ensure only one vendor is defined
#if defined(MEGATRACE_GPU_NVIDIA) && defined(MEGATRACE_GPU_AMD)
    #error "Cannot define both MEGATRACE_GPU_NVIDIA and MEGATRACE_GPU_AMD"
#endif

#ifdef MEGATRACE_GPU_NVIDIA
    // NVIDIA CUDA configuration
    #define MEGATRACE_GPU_VENDOR_NVIDIA
    #include <cuda_runtime.h>
    #include <cuda.h>
    
    // Type aliases for CUDA
    typedef cudaStream_t gpu_stream_t;
    typedef cudaError_t gpu_error_t;
    typedef cudaDeviceProp gpu_device_prop_t;
    
    // Error codes
    #define GPU_SUCCESS cudaSuccess
    #define GPU_ERROR_INVALID_VALUE cudaErrorInvalidValue
    #define GPU_ERROR_NOT_READY cudaErrorNotReady
    
    // Function wrappers
    #define gpuGetDevice cudaGetDevice
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuDeviceGetPCIBusId cudaDeviceGetPCIBusId
    #define gpuGetLastError cudaGetLastError
    #define gpuGetErrorString cudaGetErrorString
    
    // Stream operations
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    
    // Library names for symbol resolution
    // Note: Use conditional compilation in code instead of macro expansion
    
    #define MEGATRACE_GPU_VENDOR_NAME "NVIDIA"
    #define MEGATRACE_GPU_API_NAME "CUDA"

#elif defined(MEGATRACE_GPU_AMD)
    // AMD ROCm/HIP configuration
    #define MEGATRACE_GPU_VENDOR_AMD
    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
    
    // Type aliases for HIP
    typedef hipStream_t gpu_stream_t;
    typedef hipError_t gpu_error_t;
    typedef hipDeviceProp_t gpu_device_prop_t;
    
    // Error codes
    #define GPU_SUCCESS hipSuccess
    #define GPU_ERROR_INVALID_VALUE hipErrorInvalidValue
    #define GPU_ERROR_NOT_READY hipErrorNotReady
    
    // Function wrappers
    #define gpuGetDevice hipGetDevice
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuDeviceGetPCIBusId hipDeviceGetPCIBusId
    #define gpuGetLastError hipGetLastError
    #define gpuGetErrorString hipGetErrorString
    
    // Stream operations
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    
    // Library names for symbol resolution
    // Note: Use conditional compilation in code instead of macro expansion
    
    #define MEGATRACE_GPU_VENDOR_NAME "AMD"
    #define MEGATRACE_GPU_API_NAME "HIP"

#else
    #error "Unknown GPU vendor configuration"
#endif

// Helper macros for conditional compilation
#define IF_NVIDIA(...) __VA_ARGS__
#define IF_AMD(...)

#ifdef MEGATRACE_GPU_NVIDIA
    #undef IF_NVIDIA
    #undef IF_AMD
    #define IF_NVIDIA(...) __VA_ARGS__
    #define IF_AMD(...)
#elif defined(MEGATRACE_GPU_AMD)
    #undef IF_NVIDIA
    #undef IF_AMD
    #define IF_NVIDIA(...)
    #define IF_AMD(...) __VA_ARGS__
#endif

// Utility function to get PCI bus ID (vendor-agnostic)
static inline int megatrace_get_pci_bus_id(char* pci_buf, size_t buf_size) {
    int dev = -1;
    gpu_error_t err = gpuGetDevice(&dev);
    if (err != GPU_SUCCESS || dev < 0) {
        snprintf(pci_buf, buf_size, "unknown");
        return -1;
    }
    
    err = gpuDeviceGetPCIBusId(pci_buf, buf_size, dev);
    if (err != GPU_SUCCESS) {
        snprintf(pci_buf, buf_size, "unknown");
        return -1;
    }
    return 0;
}

#endif // GPU_CONFIG_H

