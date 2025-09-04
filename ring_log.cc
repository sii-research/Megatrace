//#include "nccl.h"
#define MEGA_CC
#include "ring_log.h"
#include "log.h"
//#include "core.h"
#include <sys/un.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <chrono>

const int nccl_megatrace_enable = getenv("NCCL_MEGATRACE_ENABLE") ? atoi(getenv("NCCL_MEGATRACE_ENABLE")) : 1;
const char* nccl_megatrace_log_path = getenv("NCCL_MEGATRACE_ENABLE") ? getenv("NCCL_MEGATRACE_LOG_PATH") : "./logs";
const int nccl_sensitive_time = getenv("NCCL_MEGATRACE_SENSTIME") ? atoi(getenv("NCCL_MEGATRACE_SENSTIME")) : 3000;


int64_t current_time_in_ms() {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);  // 获取当前时间

    // 将秒转换为毫秒并加上纳秒部分
    return now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

// 初始化环形缓冲区
void ring_buffer_init(ring_buffer_t *rb) {
    rb->head.store(0);
    rb->tail.store(0);
    ring_nccl_log.live = 1 ;

}
/*  * 获取环形缓冲区中当前未消费的日志数量  */
int ring_buffer_count(ring_buffer_t *rb) {
    int tail = rb->tail.load(std::memory_order_acquire);
    int head = rb->head.load(std::memory_order_acquire);
    if (head >= tail) {
        return head - tail;
    } else {
        return RING_BUFFER_SIZE - tail + head;
    }
}

/*
* 向环形缓冲区中写入一条日志消息
* 返回 0 表示写入成功，-1 表示缓冲区已满（日志丢弃）
*/
int ring_buffer_push(ring_buffer_t *rb, const char *msg) {
    int head = rb->head.load(std::memory_order_relaxed);
    int next_head = (head + 1) % RING_BUFFER_SIZE;
    int tail = rb->tail.load(std::memory_order_acquire);
    if (next_head == tail) {         // 缓冲区满
        tail = (tail + 1) % RING_BUFFER_SIZE;
    }
    std::string msg_str(msg);  // 将 msg 转换为 std::string
    if (msg_str.length() < LOG_MAX_LEN) {
        std::strcpy(rb->buffer[head].msg, msg_str.c_str());
    } else {
        std::strncpy(rb->buffer[head].msg, msg_str.c_str(), LOG_MAX_LEN - 1);
        rb->buffer[head].msg[LOG_MAX_LEN - 1] = '\0';  // 确保终止符
    }
    rb->last_write_ts.store(current_time_in_ms());
    rb->head.store(next_head,std::memory_order_release);
    rb->tail.store(tail, std::memory_order_release);
    return 0;
}
/*
* 批量从环形缓冲区中读取日志条目
* 参数 max_entries 表示最多读取的条数，将日志存入 out_entries 数组中，
* 返回实际读取的日志条数。
*/
int ring_buffer_pop_batch(ring_buffer_t *rb, log_entry_t *out_entries, int max_entries) {
    int tail = rb->tail.load(std::memory_order_relaxed);
    int head = rb->head.load(std::memory_order_acquire);
    int count;
         if (head >= tail) {
             count = head - tail;
         } else {
             count = RING_BUFFER_SIZE - tail + head;
         }
         if (count > max_entries) {
             count = max_entries;
         }
         for (int i = 0; i < count; i++) {
             int index = (tail + i) % RING_BUFFER_SIZE;
             out_entries[i] = rb->buffer[index];
         }
         rb->tail.store((tail + count) % RING_BUFFER_SIZE,std::memory_order_release);
    	 return count;
}


 /*
 * 日志写入线程：负责将环形缓冲区中的日志写入到文件中。
 * 刷新策略：  * 1. 如果缓冲区中日志数量达到 BATCH_SIZE，则立即写入。
 * 2. 如果日志数量不足，但距离上次刷新超过 FLUSH_INTERVAL_US，则写入所有已有日志。  */
void *log_writer_thread(void *arg) {
    const char *rank_str = getenv("OMPI_COMM_WORLD_RANK");
    if (rank_str == NULL) {
        LOG_ERROR_SIMPLE("Environment variable 'OMPI_COMM_RANK' not found.");
        return NULL;
    }
    int rank = atoi(rank_str); // 将 rank 从字符串转换为整数
    // 定义文件路径
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/rank_%d.log", nccl_megatrace_log_path, rank);

    // 打开文件
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        LOG_ERROR_SIMPLE("open file error, file path may not exist. errno=%d msg=%s", errno, strerror(errno));
        return NULL;
    }
    if(rank == 0){ 
	 LOG_INFO_SIMPLE("[Megatrace] start log thread.");
    }
    log_entry_t logs[BATCH_SIZE];
    int save_iter=0;
    while (1) {
        int64_t now = current_time_in_ms();
        int64_t last = ring_nccl_log.last_write_ts.load();
        int64_t time_diff = now - last;
        int num_logs = ring_buffer_count(&ring_nccl_log);   
	    if (time_diff < nccl_sensitive_time || num_logs == 0) {       
            LOG_DEBUG("time_diff: %ld  num_logs: %d",time_diff,num_logs);
            sleep(1);  
        } else {            
                 save_iter++;
                 LOG_INFO("[save %d] save %d logs",save_iter,num_logs);
                 int n_logs = ring_buffer_pop_batch(&ring_nccl_log, logs, num_logs);
                 for (int i = 0; i < n_logs; i++) {
                    fprintf(fp, "[save_count %d] %s\n", save_iter,logs[i].msg);
                 }
                 fflush(fp);
                 sleep(1); 
        }
    }
    fclose(fp);
    return NULL;
}

void log_event(struct timespec time_api, size_t count, const char* opName, cudaStream_t stream,int64_t opCount,int64_t groupHash) {
    //log_event(time_api, info->count, info->opName, info->stream, info->comm->opCount,info->count,info->comm,);
    // 用于格式化日志信息
    char log_msg[LOG_MAX_LEN];
    char time_str[64];
    snprintf(time_str, sizeof(time_str), "%ld.%09ld", time_api.tv_sec, time_api.tv_nsec);
    const char *rank_str = getenv("OMPI_COMM_WORLD_RANK");
    int rank = atoi(rank_str);
    // 格式化日志内容
    snprintf(log_msg, sizeof(log_msg), "[%s] [Rank %d] Fun %s Data %zu stream %p opCount %ld groupHash %ld",
             time_str,rank ,opName, count, (void*)stream,opCount,groupHash);
    //int num = ring_buffer_count(&ring_nccl_log);
    ring_buffer_push(&ring_nccl_log, log_msg);
}
