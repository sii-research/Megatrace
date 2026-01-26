#ifndef INTERCEPT_COMMON_H
#define INTERCEPT_COMMON_H

#include "gpu_config.h"
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
#include <atomic>
#include <cstdint>
#include <dlfcn.h>

// Generic hash function for UniqueId (vendor-agnostic)
uint64_t hashUniqueId(const void* id_bytes, size_t id_size);

// Log writer thread initialization (shared)
extern std::atomic<bool> g_log_thread_initialized;
extern pthread_t g_log_thread_id;
extern std::atomic<bool> g_log_thread_running;

bool init_log_writer_thread();
#define ENSURE_LOG_THREAD_READY() \
    do { \
        if (__builtin_expect(!g_log_thread_initialized.load(std::memory_order_acquire), 0)) { \
            init_log_writer_thread(); \
        } \
    } while(0)

// Generic symbol resolution function
template<typename FuncPtrType>
FuncPtrType resolve_symbol_common(const char* symbol_name, const char* const* library_names) {
    // First try RTLD_NEXT (searches libraries loaded after this one)
    void* sym = dlsym(RTLD_NEXT, symbol_name);
    if (sym != nullptr) {
        return reinterpret_cast<FuncPtrType>(sym);
    }
    // Clear any error from dlsym before trying next strategy
    dlerror();
    
    // Try to find symbol in specified libraries
    if (library_names != nullptr) {
        for (int i = 0; library_names[i] != nullptr; i++) {
            // Try already-loaded libraries first (RTLD_NOLOAD)
            void* handle = dlopen(library_names[i], RTLD_LAZY | RTLD_NOLOAD);
            if (handle == nullptr) {
                handle = dlopen(library_names[i], RTLD_LAZY);
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
    }
    return nullptr;
}

// Per-stream operation counters (vendor-agnostic)
extern std::mutex g_stream_opcount_mutex;
extern std::unordered_map<gpu_stream_t, int64_t> g_stream_to_opcount;

static inline int64_t next_opcount_for_stream(gpu_stream_t stream) {
    std::lock_guard<std::mutex> lock(g_stream_opcount_mutex);
    int64_t &counter = g_stream_to_opcount[stream];
    counter += 1;
    return counter;
}

#endif // INTERCEPT_COMMON_H

