#ifndef RING_LOG_H
#define RING_LOG_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <atomic>   
#include <cstring>
#include <unistd.h>
#include "gpu_config.h"
#include <atomic>

#define RING_BUFFER_SIZE 10000  // Ring buffer capacity
#define LOG_MAX_LEN 256       // Max length of a log entry
#define BATCH_SIZE        10240      // Entries processed per batch in the worker thread
#define FLUSH_INTERVAL_MS 4000 // Periodic flush interval in milliseconds (4s)
#define MEGATRACE_LOG_ENABLE    1
#define LOG_ROTATE_SIZE 1024 * 1024 // 1MB
#define MAX_LOG_VERSIONS 3  // Number of rotated log files to keep

extern const int nccl_megatrace_enable;
extern const char* nccl_megatrace_log_path;
// Control whether cudaStreamWaitEvent intercept logging is enabled.
// 0 = disabled, 1 = enabled
extern const int stream_wait_enable;

// Log entry structure
typedef struct {
    char msg[LOG_MAX_LEN];
    char type;
} log_entry_t;

// Ring buffer with atomics for thread safety
typedef struct {
    log_entry_t buffer[RING_BUFFER_SIZE];
    pthread_t thread;
    volatile int live = -1;
    std::atomic<int64_t> last_write_ts;
    std::atomic<int> head;  // Write pointer (producer)
    std::atomic<int> tail;  // Read pointer (consumer)
} ring_buffer_t;

void ring_buffer_init(ring_buffer_t *rb) ;
int ring_buffer_count(ring_buffer_t *rb) ;
int ring_buffer_push(ring_buffer_t *rb, const char *msg);
int ring_buffer_pop_batch(ring_buffer_t *rb, log_entry_t *out_entries, int max_entries) ;
void *log_writer_thread(void *arg) ;
void log_event(struct timespec time_api, size_t count, const char* opName, gpu_stream_t stream, int64_t opCount, uint64_t groupHash);

#ifdef MEGA_CC
ring_buffer_t ring_nccl_log;
#else
extern ring_buffer_t ring_nccl_log;
#endif
#endif
