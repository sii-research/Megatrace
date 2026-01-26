#ifdef MEGATRACE_GPU_AMD

#include "intercept_common.h"
#include "intercept.h"
#include "ring_log.h"
#include "log.h"
#include <pthread.h>
#include <time.h>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <stack>

// Define function pointer variables declared in intercept.h
// Note: RCCL may use different function names or maintain NCCL compatibility
// We'll try both naming conventions
ncclReduce_t real_ncclReduce = NULL;
ncclBroadcast_t real_ncclBroadcast = NULL;
ncclAllReduce_t real_ncclAllReduce = NULL;
ncclReduceScatter_t real_ncclReduceScatter = NULL;
ncclAllGather_t real_ncclAllGather = NULL;
ncclSendRecv_t real_ncclSendRecv = NULL;
ncclSend_t real_ncclSend = NULL;
ncclRecv_t real_ncclRecv = NULL;
ncclCommInitRank_t real_ncclCommInitRank = NULL;
ncclCommInitRankConfig_t real_ncclCommInitRankConfig = NULL;
ncclCommInitAll_t real_ncclCommInitAll = NULL;
ncclGetUniqueId_t real_ncclGetUniqueId = NULL;

// Mutex for thread-safe dlsym operations
static std::mutex dlsym_mutex;

// Custom hash and equality functions for RCCL (using ncclUniqueId type for compatibility)
struct RcclUniqueIdHash {
    std::size_t operator()(const ncclUniqueId& id) const {
        return static_cast<std::size_t>(hashUniqueId(&id, sizeof(ncclUniqueId)));
    }
};

struct RcclUniqueIdEqual {
    bool operator()(const ncclUniqueId& lhs, const ncclUniqueId& rhs) const {
        return memcmp(&lhs, &rhs, sizeof(ncclUniqueId)) == 0;
    }
};

// Global mapping tables using custom hash/equality
static std::mutex g_comm_mapping_mutex;
static std::unordered_map<ncclUniqueId, ncclComm_t, RcclUniqueIdHash, RcclUniqueIdEqual> g_comm_id_to_comm;
static std::unordered_map<ncclComm_t, ncclUniqueId, std::hash<void*>, std::equal_to<void*>> g_comm_to_comm_id;
// Use thread-local stack to support repeated calls
static thread_local std::stack<ncclUniqueId> g_unique_id_stack;

// Helper: fetch commId by comm
ncclUniqueId getCommIdByComm(ncclComm_t comm) {
    std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
    auto it = g_comm_to_comm_id.find(comm);
    if (it != g_comm_to_comm_id.end()) {
        return it->second;
    }
    // Return an empty commId (all zeros)
    ncclUniqueId emptyId = {0};
    return emptyId;
}

// Helper: fetch comm by commId
ncclComm_t getCommByCommId(ncclUniqueId commId) {
    std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
    auto it = g_comm_id_to_comm.find(commId);
    if (it != g_comm_id_to_comm.end()) {
        return it->second;
    }
    return nullptr;
}

uint64_t hashUniqueIdTyped(ncclUniqueId const &id) {
    return hashUniqueId(static_cast<const void*>(&id), sizeof(ncclUniqueId));
}

uint64_t getGroupHash(ncclComm_t comm) {
    if (comm == nullptr) {
        return 0;
    }
    // Only use mapping table: comm -> commId -> hash
    ncclUniqueId commId = getCommIdByComm(comm);
    // Check if commId is empty (all zeros)
    bool isEmpty = true;
    const char* bytes = reinterpret_cast<const char*>(&commId);
    for(int i = 0; i < (int)sizeof(ncclUniqueId); i++) {
        if (bytes[i] != 0) {
            isEmpty = false;
            break;
        }
    }
    if (isEmpty) return 0;
    // Use our stable hash of commId
    return hashUniqueIdTyped(commId);
}

// Helper function to resolve symbols with multiple strategies
// Tries RTLD_NEXT first, then searches in PyTorch libraries
// RCCL may use nccl* function names for compatibility or rccl* names
template<typename FuncPtrType>
FuncPtrType resolve_symbol(const char* symbol_name) {
    // RCCL-specific library search list
    const char* torch_libs[] = {
        "libtorch_hip.so",
        "libtorch_hip.so.1",
        "libtorch_hip.so.2",
        "libtorch_python.so",
        "libtorch_python.so.1",
        "libtorch_python.so.2",
        "librccl.so.2",
        "librccl.so.3",
        "librccl.so",
        nullptr
    };
    
    // First try with original symbol name (RCCL may maintain NCCL compatibility)
    FuncPtrType result = resolve_symbol_common<FuncPtrType>(symbol_name, torch_libs);
    if (result != nullptr) {
        return result;
    }
    
    // If not found, try with rccl prefix (if RCCL uses different naming)
    // For example: ncclAllReduce -> rcclAllReduce
    if (strncmp(symbol_name, "nccl", 4) == 0) {
        char rccl_name[256];
        snprintf(rccl_name, sizeof(rccl_name), "rccl%s", symbol_name + 4);
        result = resolve_symbol_common<FuncPtrType>(rccl_name, torch_libs);
        if (result != nullptr) {
            return result;
        }
    }
    
    return nullptr;
}

// Intercept ncclGetUniqueId to capture commId
// RCCL may use ncclGetUniqueId for compatibility or rcclGetUniqueId
extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!real_ncclGetUniqueId)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclGetUniqueId) {
            real_ncclGetUniqueId = resolve_symbol<ncclGetUniqueId_t>("ncclGetUniqueId");
            if (!real_ncclGetUniqueId) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclGetUniqueId: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    ncclResult_t result = real_ncclGetUniqueId(uniqueId);
    if (result == ncclSuccess && uniqueId) {
        // Push uniqueId into thread-local stack to support repeated calls
        g_unique_id_stack.push(*uniqueId);
    }
    return result;
}

// Intercept ncclCommInitRank to map comm <-> commId
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    if (!real_ncclCommInitRank)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclCommInitRank) {
            real_ncclCommInitRank = resolve_symbol<ncclCommInitRank_t>("ncclCommInitRank");
            if (!real_ncclCommInitRank) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclCommInitRank: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    ncclResult_t result = real_ncclCommInitRank(comm, nranks, commId, rank);
    
    // On success, record mapping
    if (result == ncclSuccess && comm && *comm) {
        {
            std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
            g_comm_id_to_comm[commId] = *comm;
            g_comm_to_comm_id[*comm] = commId;
        }
        // If the top stack element matches, pop it to avoid reuse by ncclCommInitAll
        if (!g_unique_id_stack.empty() && RcclUniqueIdEqual()(g_unique_id_stack.top(), commId)) {
            g_unique_id_stack.pop();
        }
    }
    return result;
}

// Intercept ncclCommInitRankConfig to map comm <-> commId
extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
    if (!real_ncclCommInitRankConfig)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclCommInitRankConfig) {
            real_ncclCommInitRankConfig = resolve_symbol<ncclCommInitRankConfig_t>("ncclCommInitRankConfig");
            if (!real_ncclCommInitRankConfig) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclCommInitRankConfig: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }

    ncclResult_t result = real_ncclCommInitRankConfig(comm, nranks, commId, rank, config);
    
    // On success, record mapping
    if (result == ncclSuccess && comm && *comm) {
        {
            std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
            g_comm_id_to_comm[commId] = *comm;
            g_comm_to_comm_id[*comm] = commId;
        }
        // If the top stack element matches, pop it to avoid reuse by ncclCommInitAll
        if (!g_unique_id_stack.empty() && RcclUniqueIdEqual()(g_unique_id_stack.top(), commId)) {
            g_unique_id_stack.pop();
        }
    }
    
    return result;
}

// Intercept ncclCommInitAll for single-process multi-GPU scenarios
extern "C" ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
    if (!real_ncclCommInitAll)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclCommInitAll) {
            real_ncclCommInitAll = resolve_symbol<ncclCommInitAll_t>("ncclCommInitAll");
            if (!real_ncclCommInitAll) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclCommInitAll: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    ncclResult_t result = real_ncclCommInitAll(comm, ndev, devlist);
    
    // After initialization: build global mapping (without touching comm struct)
    if (result == ncclSuccess && comm) {
        if (!g_unique_id_stack.empty()) {
            ncclUniqueId commId = g_unique_id_stack.top();
            g_unique_id_stack.pop();  // Pop the used uniqueId
            {
                std::lock_guard<std::mutex> lock(g_comm_mapping_mutex);
                // Build bidirectional mapping for each comm
                for (int i = 0; i < ndev; i++) {
                    if (comm[i] != nullptr) {
                        g_comm_id_to_comm[commId] = comm[i];
                        g_comm_to_comm_id[comm[i]] = commId;
                    }
                }
            }
        } else {
            LOG_ERROR("[MEGATRACE] ncclCommInitAll called but no uniqueId found in stack. Mapping may be incomplete.");
        }
    }
    
    return result;
}

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclAllReduce)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclAllReduce) {
            real_ncclAllReduce = resolve_symbol<ncclAllReduce_t>("ncclAllReduce");
            if (!real_ncclAllReduce) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclAllReduce: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclAllReduce", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclReduceScatter)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclReduceScatter) {
            real_ncclReduceScatter = resolve_symbol<ncclReduceScatter_t>("ncclReduceScatter");
            if (!real_ncclReduceScatter) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclReduceScatter: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclReduceScatter", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclAllGather)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclAllGather) {
            real_ncclAllGather = resolve_symbol<ncclAllGather_t>("ncclAllGather");
            if (!real_ncclAllGather) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclAllGather: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclAllGather", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclSendRecv(const void* sendbuff, size_t sendcount, ncclDataType_t sendtype, int peer_send, void* recvbuff, size_t recvcount, ncclDataType_t recvtype, int peer_recv, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclSendRecv)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclSendRecv) {
            real_ncclSendRecv = resolve_symbol<ncclSendRecv_t>("ncclSendRecv");
            if (!real_ncclSendRecv) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclSendRecv: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, sendcount, "ncclSendRecv", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclSendRecv(sendbuff, sendcount, sendtype, peer_send, recvbuff, recvcount, recvtype, peer_recv, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclSend)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclSend) {
            real_ncclSend = resolve_symbol<ncclSend_t>("ncclSend");
            if (!real_ncclSend) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclSend: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclSend", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclRecv)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclRecv) {
            real_ncclRecv = resolve_symbol<ncclRecv_t>("ncclRecv");
            if (!real_ncclRecv) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclRecv: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    // Call log_event with function-specific parameters
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclRecv", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclReduce)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclReduce) {
            real_ncclReduce = resolve_symbol<ncclReduce_t>("ncclReduce");
            if (!real_ncclReduce) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclReduce: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
        int64_t opCount = next_opcount_for_stream(stream);
        log_event(ts, count, "ncclReduce", stream, opCount, groupHash);
    }
    
    ncclResult_t result = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    return result;
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, gpu_stream_t stream) {
    if (!real_ncclBroadcast)
    {
        std::lock_guard<std::mutex> lock(dlsym_mutex);
        if (!real_ncclBroadcast) {
            real_ncclBroadcast = resolve_symbol<ncclBroadcast_t>("ncclBroadcast");
            if (!real_ncclBroadcast) {
                const char* err = dlerror();
                LOG_ERROR("Cannot find symbol ncclBroadcast: %s", err ? err : "unknown error");
                return ncclSystemError;
            }
        }
    }
    
    // Ensure log writer thread is running before logging
    ENSURE_LOG_THREAD_READY();
    
    if (nccl_megatrace_enable == MEGATRACE_LOG_ENABLE) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // Derive groupHash from commId to avoid touching external ncclComm
        uint64_t groupHash = getGroupHash(comm);
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
    LOG_INFO("Megatrace RCCL interceptor initialized");
}

#endif // MEGATRACE_GPU_AMD

