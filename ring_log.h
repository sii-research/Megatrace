#ifndef RING_LOG_H
#define RING_LOG_H

#include <stdlib.h>
#include <stdio.h>
#include <timer.h>
#include <time.h>
#include <sys/types.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include "socket.h"
#include "nccl.h"
#include "core.h"
#include <atomic>   
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#include <atomic>


#define RING_BUFFER_SIZE 10000  // 环形缓冲区的大小
#define LOG_MAX_LEN 128       // 日志条目大小
#define BATCH_SIZE        10240      // 子线程每次批量处理日志的条数
#define FLUSH_INTERVAL_MS 4000 // 定时刷新间隔（单位：微秒，这里设置为2s）
#define MEGATRACE_LOG_ENABLE           1
//#define NCCL_COLL_LOG 0
//#define NCCL_TELEMERTRY_LOG 1
extern const int nccl_megatrace_enable;
extern const char* nccl_megatrace_log_path;



// 日志条目结构体
typedef struct {
    char msg[LOG_MAX_LEN];
    char type;
} log_entry_t;


// 环形缓冲区结构体，采用原子变量确保多线程安全
typedef struct {
    log_entry_t buffer[RING_BUFFER_SIZE];
    pthread_t thread;
    volatile int live = -1;
    std::atomic<int64_t> last_write_ts;
    std::atomic<int> head;  // 写指针（生产者更新）
    std::atomic<int> tail;  // 读指针（消费者更新）
} ring_buffer_t;



void ring_buffer_init(ring_buffer_t *rb) ;
int ring_buffer_count(ring_buffer_t *rb) ;
int ring_buffer_push(ring_buffer_t *rb, const char *msg);
int ring_buffer_pop_batch(ring_buffer_t *rb, log_entry_t *out_entries, int max_entries) ;
void *log_writer_thread(void *arg) ;
void log_event(struct timespec time_api, size_t count, const char* opName, cudaStream_t stream,int64_t opCount,int64_t groupHash);



#ifdef MEGA_CC
ring_buffer_t ring_nccl_log;
#else
extern ring_buffer_t ring_nccl_log;
#endif


#endif
