#ifndef INTERCEPT_H
#define INTERCEPT_H
#include <cuda_runtime.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <dlfcn.h>

// cuda_intercept
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


// nccl_intercept
typedef ncclResult_t  (*ncclReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t  (*ncclBroadcast_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t (*ncclAllReduce_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduceScatter_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclAllGather_t)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSendRecv_t)(const void*, size_t, ncclDataType_t, int, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSend_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t (*ncclRecv_t)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);

static ncclReduce_t real_ncclReduce = NULL;
static ncclBroadcast_t real_ncclBroadcast = NULL;
static ncclAllReduce_t real_ncclAllReduce = NULL;
static ncclReduceScatter_t real_ncclReduceScatter = NULL;
static ncclAllGather_t real_ncclAllGather = NULL;
static ncclSendRecv_t real_ncclSendRecv = NULL;
static ncclSend_t real_ncclSend = NULL;
static ncclRecv_t real_ncclRecv = NULL;

#endif
