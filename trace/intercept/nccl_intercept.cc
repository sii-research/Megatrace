#include <nccl.h>
#include <dlfcn.h>
#include <algorithm>
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
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <stack>

// Global variables for log writer thread initialization control
static std::atomic<bool> g_log_thread_initialized{false};
static pthread_t g_log_thread_id = 0;
static std::atomic<bool> g_log_thread_running{false};

// Define function pointer variables declared in intercept.h
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

// Per-stream operation counters
static std::mutex g_stream_opcount_mutex;
static std::unordered_map<cudaStream_t, int64_t> g_stream_to_opcount;

static inline int64_t next_opcount_for_stream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_stream_opcount_mutex);
    int64_t &counter = g_stream_to_opcount[stream];
    counter += 1;
    return counter;
}

// Forward declaration for NcclUniqueIdHash
uint64_t hashUniqueId(ncclUniqueId const &id);

// Custom hash and equality functions
struct NcclUniqueIdHash {
    std::size_t operator()(const ncclUniqueId& id) const {
        return static_cast<std::size_t>(hashUniqueId(id));
    }
};

struct NcclUniqueIdEqual {
    bool operator()(const ncclUniqueId& lhs, const ncclUniqueId& rhs) const {
        return memcmp(&lhs, &rhs, sizeof(ncclUniqueId)) == 0;
    }
};

// Global mapping tables using custom hash/equality
static std::mutex g_comm_mapping_mutex;
static std::unordered_map<ncclUniqueId, ncclComm_t, NcclUniqueIdHash, NcclUniqueIdEqual> g_comm_id_to_comm;
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

// From NCCL bitops.h - getHash implementation
static uint64_t getHash(const void* bytes, size_t size) {
    uint64_t h = 0xdeadbeef;
    char const *ptr = static_cast<char const*>(bytes);
    for (size_t i = 0; i < size; i++) {
        h ^= h >> 32;
        h *= 0x8db3db47fa2994adULL;
        h += ptr[i];
    }
    return h;
}

uint64_t getHash(const char* bytes, int n) {
    return getHash(static_cast<const void*>(bytes), static_cast<size_t>(n));
}

uint64_t hashUniqueId(ncclUniqueId const &id) {
    return getHash(static_cast<const void*>(&id), sizeof(ncclUniqueId));
}

uint64_t getGroupHash(ncclComm_t comm) {
    if (comm == nullptr) {
        return 0;
    }
    // Only use mapping table: comm -> commId -> hash
    ncclUniqueId commId = getCommIdByComm(comm);
    // Check if commId is empty (all zeros)
    bool isEmpty = true;
    char const *bytes = (char const*)&commId;
    for(int i = 0; i < (int)sizeof(ncclUniqueId); i++) {
        if (bytes[i] != 0) {
            isEmpty = false;
            break;
        }
    }
    if (isEmpty) return 0;
    // Use our stable hash of commId
    return hashUniqueId(commId);
}

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

// Helper function to resolve symbols with multiple strategies
// Tries RTLD_NEXT first, then searches in PyTorch libraries, then RTLD_DEFAULT with verification
template<typename FuncPtrType>
FuncPtrType resolve_symbol(const char* symbol_name) {
    // First try RTLD_NEXT (searches libraries loaded after this one)
    // This is the safest option as it won't find our own function
    void* sym = dlsym(RTLD_NEXT, symbol_name);
    if (sym != nullptr) {
        return reinterpret_cast<FuncPtrType>(sym);
    }
    // Clear any error from dlsym before trying next strategy
    dlerror();
    // Try to find symbol in PyTorch libraries (NCCL might be loaded via dlopen by PyTorch)
    // Common PyTorch library names that might contain NCCL
    const char* torch_libs[] = {
        "libtorch_cuda.so",
        "libtorch_cuda.so.1",
        "libtorch_cuda.so.2",
        "libtorch_python.so",
        "libtorch_python.so.1",
        "libtorch_python.so.2",
        "libnccl.so.2",
        "libnccl.so.3",
        "libnccl.so",
        nullptr
    };
    for (int i = 0; torch_libs[i] != nullptr; i++) {
        // Try already-loaded libraries first (RTLD_NOLOAD)
        void* handle = dlopen(torch_libs[i], RTLD_LAZY | RTLD_NOLOAD);
        if (handle == nullptr) {
            handle = dlopen(search_libs[i], RTLD_LAZY);
        }
        if (handle != nullptr) {
            sym = dlsym(handle, symbol_name);
            if (sym != nullptr) {
                return reinterpret_cast<FuncPtrType>(sym);
            }
            dlclose(handle);
        }
        dlerror(); // Clear error
    }
    return nullptr;
}

// Intercept ncclGetUniqueId to capture commId
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
        if (!g_unique_id_stack.empty() && NcclUniqueIdEqual()(g_unique_id_stack.top(), commId)) {
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
        if (!g_unique_id_stack.empty() && NcclUniqueIdEqual()(g_unique_id_stack.top(), commId)) {
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

extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclSendRecv(const void* sendbuff, size_t sendcount, ncclDataType_t sendtype, int peer_send, void* recvbuff, size_t recvcount, ncclDataType_t recvtype, int peer_recv, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
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

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
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
    LOG_INFO("Megatrace CCL interceptor initialized");
}
