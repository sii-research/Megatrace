#include "intercept_common.h"
#include "ring_log.h"
#include <errno.h>

// Global log writer thread control variables
std::atomic<bool> g_log_thread_initialized{false};
pthread_t g_log_thread_id = 0;
std::atomic<bool> g_log_thread_running{false};

// Per-stream operation counters
std::mutex g_stream_opcount_mutex;
std::unordered_map<gpu_stream_t, int64_t> g_stream_to_opcount;

// Generic hash function implementation (from NCCL bitops.h)
uint64_t hashUniqueId(const void* bytes, size_t size) {
    uint64_t h = 0xdeadbeef;
    const char* ptr = static_cast<const char*>(bytes);
    for (size_t i = 0; i < size; i++) {
        h ^= h >> 32;
        h *= 0x8db3db47fa2994adULL;
        h += ptr[i];
    }
    return h;
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

