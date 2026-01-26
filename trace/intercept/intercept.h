#ifndef INTERCEPT_H
#define INTERCEPT_H
#include "gpu_config.h"
#include <dlfcn.h>

#ifdef MEGATRACE_GPU_NVIDIA
    #include <nccl.h>
    #include <cublas_v2.h>
    // CUDA intercept function pointers
    typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t, cudaEvent_t, unsigned int);
    typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t, cudaStream_t);
    typedef cudaError_t (*cudaEventQuery_t)(cudaEvent_t);
    typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t);
    typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    typedef cudaError_t (*real_cudaFuncGetAttributes_t)(struct cudaFuncAttributes *, const void *);
    typedef cudaError_t (*real_cudaMemcpyAsync_t)(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);

    static real_cudaFuncGetAttributes_t real_cudaFuncGetAttributes = NULL;
    static real_cudaMemcpyAsync_t real_cudaMemcpyAsync = NULL;
    static cudaStreamWaitEvent_t real_cudaStreamWaitEvent = nullptr;
    static cudaEventRecord_t real_cudaEventRecord = nullptr;
    static cudaEventQuery_t real_cudaEventQuery = nullptr;
    static cudaEventDestroy_t  real_cudaEventDestroy = nullptr;
    static cudaLaunchKernel_t real_cudaLaunchKernel = NULL;
    
    // NCCL types (native)
    typedef ncclComm_t comm_t;
    typedef ncclUniqueId unique_id_t;
    typedef ncclResult_t result_t;
    
#elif defined(MEGATRACE_GPU_AMD)
    // RCCL header - check if it provides nccl.h compatibility or uses rccl.h
    #ifdef RCCL_USE_NCCL_COMPAT
        #include <nccl.h>
        // RCCL may provide compatibility layer using NCCL types
        typedef ncclComm_t comm_t;
        typedef ncclUniqueId unique_id_t;
        typedef ncclResult_t result_t;
    #else
        #include <rccl.h>
        // RCCL native types (if different from NCCL)
        // Note: RCCL typically maintains API compatibility, but we handle both cases
        typedef rcclComm_t comm_t;
        typedef rcclUniqueId unique_id_t;
        typedef rcclResult_t result_t;
        
        // Type aliases for compatibility (if RCCL uses different names)
        #ifndef ncclComm_t
            typedef rcclComm_t ncclComm_t;
        #endif
        #ifndef ncclUniqueId
            typedef rcclUniqueId ncclUniqueId;
        #endif
        #ifndef ncclResult_t
            typedef rcclResult_t ncclResult_t;
        #endif
        #ifndef ncclDataType_t
            typedef rcclDataType_t ncclDataType_t;
        #endif
        #ifndef ncclRedOp_t
            typedef rcclRedOp_t ncclRedOp_t;
        #endif
        #ifndef ncclConfig_t
            typedef rcclConfig_t ncclConfig_t;
        #endif
        
        // Result constants
        #ifndef ncclSuccess
            #define ncclSuccess rcclSuccess
        #endif
        #ifndef ncclSystemError
            #define ncclSystemError rcclSystemError
        #endif
    #endif
#else
    #error "Unknown GPU vendor - must define MEGATRACE_GPU_NVIDIA or MEGATRACE_GPU_AMD"
#endif

// Collective communication intercept function pointers
// Use vendor-agnostic types where possible, but maintain compatibility
typedef result_t  (*commReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, comm_t comm, gpu_stream_t stream);
typedef result_t  (*commBroadcast_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, comm_t comm, gpu_stream_t stream);
typedef result_t (*commAllReduce_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, comm_t, gpu_stream_t);
typedef result_t (*commReduceScatter_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, comm_t, gpu_stream_t);
typedef result_t (*commAllGather_t)(const void*, void*, size_t, ncclDataType_t, comm_t, gpu_stream_t);
typedef result_t (*commSendRecv_t)(const void*, size_t, ncclDataType_t, int, void*, size_t, ncclDataType_t, int, comm_t, gpu_stream_t);
typedef result_t (*commSend_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, comm_t comm, gpu_stream_t stream);
typedef result_t (*commRecv_t)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, comm_t comm, gpu_stream_t stream);
typedef result_t (*commInitRank_t)(comm_t* comm, int nranks, unique_id_t commId, int rank);
typedef result_t (*commInitRankConfig_t)(comm_t* comm, int nranks, unique_id_t commId, int rank, ncclConfig_t* config);
typedef result_t (*commInitAll_t)(comm_t* comm, int ndev, const int* devlist);
typedef result_t (*commGetUniqueId_t)(unique_id_t* uniqueId);

// NCCL-compatible function pointer types (for backward compatibility and API compatibility)
typedef result_t  (*ncclReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, gpu_stream_t stream);
typedef result_t  (*ncclBroadcast_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, gpu_stream_t stream);
typedef result_t (*ncclAllReduce_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, gpu_stream_t);
typedef result_t (*ncclReduceScatter_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, gpu_stream_t);
typedef result_t (*ncclAllGather_t)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, gpu_stream_t);
typedef result_t (*ncclSendRecv_t)(const void*, size_t, ncclDataType_t, int, void*, size_t, ncclDataType_t, int, ncclComm_t, gpu_stream_t);
typedef result_t (*ncclSend_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, gpu_stream_t stream);
typedef result_t (*ncclRecv_t)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, gpu_stream_t stream);
typedef result_t (*ncclCommInitRank_t)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
typedef result_t (*ncclCommInitRankConfig_t)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
typedef result_t (*ncclCommInitAll_t)(ncclComm_t* comm, int ndev, const int* devlist);
typedef result_t (*ncclGetUniqueId_t)(ncclUniqueId* uniqueId);

#endif
