#define MEGA_CC
#include "ring_log.h"
#include "log.h"
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
    clock_gettime(CLOCK_REALTIME, &now);  // Get current time
    return now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

// Initialize ring buffer
void ring_buffer_init(ring_buffer_t *rb) {
    rb->head.store(0);
    rb->tail.store(0);
    ring_nccl_log.live = 1 ;

}
/*  * Get number of unread log entries in the ring buffer  */
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
* Write a log entry into the ring buffer.
* Returns 0 on success; drop oldest entry when the buffer is full.
*/
int ring_buffer_push(ring_buffer_t *rb, const char *msg) {
    int head = rb->head.load(std::memory_order_relaxed);
    int next_head = (head + 1) % RING_BUFFER_SIZE;
    int tail = rb->tail.load(std::memory_order_acquire);
    if (next_head == tail) {         // Buffer full, advance tail to drop oldest
        tail = (tail + 1) % RING_BUFFER_SIZE;
    }
    std::string msg_str(msg);  // Convert msg to std::string
    if (msg_str.length() < LOG_MAX_LEN) {
        std::strcpy(rb->buffer[head].msg, msg_str.c_str());
    } else {
        std::strncpy(rb->buffer[head].msg, msg_str.c_str(), LOG_MAX_LEN - 1);
        rb->buffer[head].msg[LOG_MAX_LEN - 1] = '\0';  // Ensure string terminator
    }
    rb->last_write_ts.store(current_time_in_ms());
    rb->head.store(next_head,std::memory_order_release);
    rb->tail.store(tail, std::memory_order_release);
    return 0;
}
/*
* Pop up to max_entries from the ring buffer into out_entries.
* Returns the number of entries copied.
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
* Rotate log files: rename existing files with version suffix and drop the oldest.
* filename: current log file (full path)
* Returns 0 on success, -1 on failure.
*/
int rotate_log_file(const char *filename) {
    char old_name[512];
    char new_name[512];
    
    // Rename from the highest version backward (e.g., .3 -> .4, .2 -> .3, .1 -> .2)
    // to avoid overwriting existing files.
    for (int i = MAX_LOG_VERSIONS - 1; i >= 1; i--) {
        snprintf(old_name, sizeof(old_name), "%s.%d", filename, i);
        snprintf(new_name, sizeof(new_name), "%s.%d", filename, i + 1);
        
        // If an older version exists, rename it to the next version
        if (access(old_name, F_OK) == 0) {
            if (rename(old_name, new_name) != 0) {
                LOG_ERROR_SIMPLE("failed to rename log file from %s to %s. errno=%d msg=%s",
                                old_name, new_name, errno, strerror(errno));
                // 继续处理其他文件，不因单个文件失败而停止
            }
        }
    }
    
    // Rename current file to .1
    snprintf(new_name, sizeof(new_name), "%s.1", filename);
    if (rename(filename, new_name) != 0) {
        LOG_ERROR_SIMPLE("failed to rename current log file to %s. errno=%d msg=%s",
                        new_name, errno, strerror(errno));
        return -1;
    }
    
    // Remove the file that exceeds MAX_LOG_VERSIONS (if present)
    snprintf(old_name, sizeof(old_name), "%s.%d", filename, MAX_LOG_VERSIONS + 1);
    if (access(old_name, F_OK) == 0) {
        if (remove(old_name) != 0) {
            LOG_ERROR_SIMPLE("failed to remove old log file %s. errno=%d msg=%s",
                            old_name, errno, strerror(errno));
        }
    }
    
    return 0;
}

// Get PCI bus ID of current CUDA device
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
 * Log writer thread: moves buffered entries to disk.
 * Flush policy: write logs when there is data and time since last write
 * exceeds nccl_sensitive_time (ms). Also rotates files after LOG_ROTATE_SIZE.
 */
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

    // Fetch hostname
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

    // Get current time string
    time_t rawtime;
    struct tm *timeinfo;
    char time_buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", timeinfo);

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s_%s_%d.log", nccl_megatrace_log_path, pod_name, time_buffer, pid);

    // Open log file
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
                fclose(fp);
                if (rotate_log_file(filename) != 0) {
                    LOG_ERROR_SIMPLE("failed to rotate log file: %s", filename);
                }
                
                fp = fopen(filename, "w");
                if (!fp) {
                    LOG_ERROR_SIMPLE("failed to create new log file after rotation. errno=%d msg=%s", 
                                     errno, strerror(errno));
                    return NULL;
                }
                
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
   
    char log_msg[LOG_MAX_LEN];
    char time_str[64];
    snprintf(time_str, sizeof(time_str), "%ld.%09ld", time_api.tv_sec, time_api.tv_nsec);
    
    // Get rank
    const char *rank_str = get_rank_str();
    int rank = (rank_str != NULL) ? atoi(rank_str) : 0;
    
    // Get PCI bus ID
    char pci_bus_id[32] = {0};
    get_pci_bus_id(pci_bus_id, sizeof(pci_bus_id));
    
    snprintf(log_msg, sizeof(log_msg), "[%s] [Rank %d] [PCI %s] [Func %s] [Data %zu] [stream %p] [opCount %lld] [groupHash 0x%016llx]",
            time_str, rank, pci_bus_id, opName, count, (void*)stream, (long long)opCount, (unsigned long long)groupHash);
    ring_buffer_push(&ring_nccl_log, log_msg);
}