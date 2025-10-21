#include <nccl.h>
#include <dlfcn.h>
#include <iostream>
#include <atomic>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "intercept.h"
#include "ring_log.h"
#include "log.h"
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include "include/comm.h"
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <mutex>

// Global variables for log writer thread initialization control
static std::atomic<bool> g_log_thread_initialized{false};
static pthread_t g_log_thread_id = 0;
static std::atomic<bool> g_log_thread_running{false};

// Per-stream operation counters
static std::mutex g_stream_opcount_mutex;
static std::unordered_map<cudaStream_t, int64_t> g_stream_to_opcount;

static inline int64_t next_opcount_for_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_stream_opcount_mutex);
    int64_t &counter = g_stream_to_opcount[stream];
    counter += 1;
    return counter;
}

// Best-effort groupHash fetcher: if env NCCL_COMM_GROUPHASH_OFFSET is set to a byte offset
// within ncclComm_t where groupHash resides, read it; otherwise fallback to pointer hash
// Directly read groupHash from ncclComm now that internal headers are available

// Initialize log writer thread - ensures only one initialization
bool init_log_writer_thread() {
    // Simple atomic check - no mutex needed for read-only check
    if (g_log_thread_initialized.load(std::memory_order_acquire)) {
        return true;
    }
    
    // Use compare_exchange_strong for atomic initialization
    bool expected = false;
    if (g_log_thread_initialized.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        // We won the race to initialize
        
        // Initialize ring buffer first
        ring_buffer_init(&ring_nccl_log);
        
        // Create log writer thread
        int ret = pthread_create(&g_log_thread_id, NULL, log_writer_thread, NULL);
        if (ret != 0) {
            LOG_ERROR("Failed to create log writer thread: %s", strerror(ret));
            g_log_thread_initialized.store(false, std::memory_order_release);
            return false;
        }
        
        // Wait a bit for thread to start
        usleep(10000); // 10ms
        
        g_log_thread_running.store(true, std::memory_order_release);
        
        LOG_INFO("Log writer thread initialized successfully");
        return true;
    } else {
        // Another thread won the race, wait for it to complete
        while (!g_log_thread_running.load(std::memory_order_acquire)) {
            // Spin briefly, then yield
            for (volatile int i = 0; i < 1000; i++) {}
            sched_yield();
        }
        return true;
    }
}

// Ultra-lightweight check - just atomic read, no function call overhead
#define ENSURE_LOG_THREAD_READY() \
    do { \
        if (__builtin_expect(!g_log_thread_initialized.load(std::memory_order_acquire), 0)) { \
            init_log_writer_thread(); \
        } \
    } while(0)

// Individual NCCL function implementations with direct parameter access

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclAllReduce) {
        real_ncclAllReduce = (ncclAllReduce_t)dlsym(handle, "ncclAllReduce");
        if (!real_ncclAllReduce) {
            LOG_ERROR("Cannot find symbol ncclAllReduce: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclAllReduce", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclReduceScatter) {
        real_ncclReduceScatter = (ncclReduceScatter_t)dlsym(handle, "ncclReduceScatter");
        if (!real_ncclReduceScatter) {
            LOG_ERROR("Cannot find symbol ncclReduceScatter: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclReduceScatter", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclAllGather) {
        real_ncclAllGather = (ncclAllGather_t)dlsym(handle, "ncclAllGather");
        if (!real_ncclAllGather) {
            LOG_ERROR("Cannot find symbol ncclAllGather: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclAllGather", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclSendRecv(const void* sendbuff, size_t sendcount, ncclDataType_t sendtype, int peer_send, void* recvbuff, size_t recvcount, ncclDataType_t recvtype, int peer_recv, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclSendRecv) {
        real_ncclSendRecv = (ncclSendRecv_t)dlsym(handle, "ncclSendRecv");
        if (!real_ncclSendRecv) {
            LOG_ERROR("Cannot find symbol ncclSendRecv: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, sendcount, "ncclSendRecv", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclSendRecv(sendbuff, sendcount, sendtype, peer_send, recvbuff, recvcount, recvtype, peer_recv, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclSend) {
        real_ncclSend = (ncclSend_t)dlsym(handle, "ncclSend");
        if (!real_ncclSend) {
            LOG_ERROR("Cannot find symbol ncclSend: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclSend", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    if (!real_ncclRecv) {
        real_ncclRecv = (ncclRecv_t)dlsym(handle, "ncclRecv");
        if (!real_ncclRecv) {
            LOG_ERROR("Cannot find symbol ncclRecv: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclRecv", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclReduce) {
        real_ncclReduce = (ncclReduce_t)dlsym(handle, "ncclReduce");
        if (!real_ncclReduce) {
            LOG_ERROR("Cannot find symbol ncclReduce: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclReduce", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    dlclose(handle);
    return result;
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!handle) {
        LOG_ERROR("Failed to dlopen libnccl.so: %s", dlerror());
        return ncclSystemError;
    }
    
    if (!real_ncclBroadcast) {
        real_ncclBroadcast = (ncclBroadcast_t)dlsym(handle, "ncclBroadcast");
        if (!real_ncclBroadcast) {
            LOG_ERROR("Cannot find symbol ncclBroadcast: %s", dlerror());
            dlclose(handle);
            return ncclSystemError;
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        int64_t groupHash = static_cast<int64_t>(reinterpret_cast<ncclComm*>(comm)->groupHash);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclBroadcast", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
    return result;
}

// Initialize logging system when library is loaded
__attribute__((constructor))
void megatrace_init() {
    log_init();
    LOG_INFO("Megatrace NCCL interceptor initialized");
}