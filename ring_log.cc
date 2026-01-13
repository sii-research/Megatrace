//#include "nccl.h"
#define MEGA_CC

#include "ring_log.h"
#include "log.h"
//#include "core.h"
#include <sys/un.h>
#include <utime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

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

std::string get_running_round(const std::string& pod_name) {
    int dash_count = std::count(pod_name.begin(), pod_name.end(), '-');
    // example pod name job-34e2a67a-d3d5-43ef-9d3a-07e9986a7355-worker-1-8
    if (dash_count < 8) {
        return "0";
    }
    size_t last_dash_pos = pod_name.find_last_of('-');
    if (last_dash_pos == std::string::npos) {
        return "0";
    }
    return pod_name.substr(last_dash_pos + 1);
}

/*
* 执行日志文件轮转：将旧文件重命名为带版本号的文件，并删除超出最大版本数的旧文件
* 参数 filename: 当前日志文件名（完整路径）
* 返回 0 表示成功，-1 表示失败
*/
int rotate_log_file(const char *filename) {
    char old_name[512];
    char new_name[512];
    
    // 从最大版本号开始，向后重命名文件（例如：.3 -> .4, .2 -> .3, .1 -> .2）
    // 这样可以避免覆盖已存在的文件
    for (int i = MAX_LOG_VERSIONS - 1; i >= 1; i--) {
        snprintf(old_name, sizeof(old_name), "%s.%d", filename, i);
        snprintf(new_name, sizeof(new_name), "%s.%d", filename, i + 1);
        
        // 如果旧版本文件存在，重命名为下一个版本号
        if (access(old_name, F_OK) == 0) {
            if (rename(old_name, new_name) != 0) {
                LOG_ERROR_SIMPLE("failed to rename log file from %s to %s. errno=%d msg=%s",
                                old_name, new_name, errno, strerror(errno));
                // 继续处理其他文件，不因单个文件失败而停止
            }
        }
    }
    
    // 将当前文件重命名为 .1
    snprintf(new_name, sizeof(new_name), "%s.1", filename);
    if (rename(filename, new_name) != 0) {
        LOG_ERROR_SIMPLE("failed to rename current log file to %s. errno=%d msg=%s",
                        new_name, errno, strerror(errno));
        return -1;
    }
    
    // 删除超出最大版本数的旧文件（如果存在）
    snprintf(old_name, sizeof(old_name), "%s.%d", filename, MAX_LOG_VERSIONS + 1);
    if (access(old_name, F_OK) == 0) {
        if (remove(old_name) != 0) {
            LOG_ERROR_SIMPLE("failed to remove old log file %s. errno=%d msg=%s",
                            old_name, errno, strerror(errno));
            // 删除失败不影响整体流程
        }
    }
    
    return 0;
}

// 获取当前CUDA设备的PCI bus ID
static void get_pci_bus_id(char* pci_buf, size_t buf_size) {
    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess || dev < 0) {
        snprintf(pci_buf, buf_size, "unknown");
        return;
    }
    
    err = cudaDeviceGetPCIBusId(pci_buf, buf_size, dev);
    if (err != cudaSuccess) {
        snprintf(pci_buf, buf_size, "unknown");
        return;
    }
}

 /*
 * 日志写入线程：负责将环形缓冲区中的日志写入到文件中。
 * 刷新策略：  * 1. 如果缓冲区中日志数量达到 BATCH_SIZE，则立即写入。
 * 2. 如果日志数量不足，但距离上次刷新超过 FLUSH_INTERVAL_US，则写入所有已有日志。  */
void *log_writer_thread(void *arg) {
    const char* pod_name = getenv("POD");
    if (pod_name == NULL) {
        pod_name = "unknown";
    }
    std::string running_round = get_running_round(pod_name);

    const char *node_ip = getenv("MY_POD_IP");
    if (node_ip == NULL) {
        node_ip = "unknown";
    }

    // 获取 hostname
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        snprintf(hostname, sizeof(hostname), "unknown");
    }

    const char* train_job_id = getenv("TRAIN_JOB_ID");
    if (train_job_id == NULL) {
        train_job_id = "unknown";
    }

    int pid = getpid();

    const char *rank_str = get_rank_str();
    int rank = (rank_str != NULL) ? atoi(rank_str) : 0;

    // 获取当前时间
    time_t rawtime;
    struct tm *timeinfo;
    char time_buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", timeinfo);

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s_%s_%d.log", nccl_megatrace_log_path, pod_name, time_buffer, pid);

    // 打开文件
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        LOG_ERROR_SIMPLE("open file error, file path may not exist. errno=%d msg=%s", errno, strerror(errno));
        return NULL;
    }
    int fd = fileno(fp);
    if (fd < 0) {
        LOG_ERROR_SIMPLE("get file descriptor error. errno=%d msg=%s", errno, strerror(errno));
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
        } else {            
            long current_pos = ftell(fp);
            if(current_pos >= 0 && current_pos >= LOG_ROTATE_SIZE) {
                // 关闭当前文件
                fclose(fp);
                
                // 执行日志轮转：重命名旧文件为带版本号的文件
                if (rotate_log_file(filename) != 0) {
                    LOG_ERROR_SIMPLE("failed to rotate log file: %s", filename);
                    // 即使轮转失败，也尝试创建新文件继续写入
                }
                
                // 创建新的同名文件
                fp = fopen(filename, "w");
                if (!fp) {
                    LOG_ERROR_SIMPLE("failed to create new log file after rotation. errno=%d msg=%s", 
                                     errno, strerror(errno));
                    return NULL;
                }
                
                // 更新文件描述符
                fd = fileno(fp);
                if (fd < 0) {
                    LOG_ERROR_SIMPLE("get file descriptor error after rotation. errno=%d msg=%s", 
                                     errno, strerror(errno));
                    fclose(fp);
                    return NULL;
                }
                
                if (rank == 0) {
                    LOG_INFO_SIMPLE("[Megatrace] log file rotated (old file renamed with version): %s", filename);
                }
            }
            save_iter++;
            LOG_INFO("[save %d] save %d logs",save_iter,num_logs);
            int n_logs = ring_buffer_pop_batch(&ring_nccl_log, logs, num_logs);
            for (int i = 0; i < n_logs; i++) {
                fprintf(fp, "[%s] [%s] [%s] [%s] [save_count %d] %s\n", train_job_id, running_round.c_str(), node_ip, hostname, save_iter, logs[i].msg);
            }
            fflush(fp);
        }
        if (save_iter % 300 == 0) {
            if (futimens(fd, NULL) == -1) {
                LOG_ERROR_SIMPLE("futimens failed =%d msg=%s", errno, strerror(errno));
            }
        }
        sleep(1); 
    }
    fclose(fp);
    return NULL;
}

void log_event(struct timespec time_api, size_t count, const char* opName, cudaStream_t stream,int64_t opCount,uint64_t groupHash) {
    //log_event(time_api, info->count, info->opName, info->stream, info->comm->opCount,info->count,info->comm,);
    // 用于格式化日志信息
    char log_msg[LOG_MAX_LEN];
    char time_str[64];
    snprintf(time_str, sizeof(time_str), "%ld.%09ld", time_api.tv_sec, time_api.tv_nsec);
    
    // 获取Rank
    const char *rank_str = get_rank_str();
    int rank = (rank_str != NULL) ? atoi(rank_str) : 0;
    
    // 获取 PCI bus ID
    char pci_bus_id[32] = {0};
    get_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));
    
    // 格式化日志内容
    snprintf(log_msg, sizeof(log_msg), "[%s] [Rank %d] [PCI %s] [Fun %s] [Data %zu] [stream %p] [opCount %lld] [groupHash 0x%016llx]",
            time_str, rank, pci_bus_id, opName, count, (void*)stream, (long long)opCount, (unsigned long long)groupHash);
    //int num = ring_buffer_count(&ring_nccl_log);
    ring_buffer_push(&ring_nccl_log, log_msg);
}