#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <memory>
#include <atomic>
#include "spdlog/include/spdlog/spdlog.h"
#include "spdlog/include/spdlog/async.h"
#include "spdlog/include/spdlog/sinks/basic_file_sink.h"
#include <unordered_map>
#include <cstdint>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>


namespace fs = std::filesystem;

const int Hang_Limit = 30;
const double Hang_Time = 30000;

// 事件信息结构
struct EventInfo {
    cudaEvent_t event;
    cudaStream_t stream;
    int Life_time;
    bool destroy;
    long long start_time;
    long long end_time;
    EventInfo(cudaEvent_t e, cudaStream_t s,long long t ) : event(e), stream(s),destroy(false),start_time(t),end_time(-1) {}
};

std::vector<EventInfo*> event_list;
std::mutex event_list_mutex;

std::unordered_set<uintptr_t> Hang_Event;

//生产者消费者队列
std::queue<std::pair<cudaEvent_t, long long> > destroy_event_queue;
std::mutex destroy_event_queue_mutex;
std::queue<EventInfo*> event_queue;
std::mutex event_queue_mutex;

std::unordered_map<uintptr_t, long long> destroy_event_list;
std::mutex destroy_event_list_mutex;

std::mutex running_mutex;
std::atomic<bool> watchdog_state(false);
std::shared_ptr<spdlog::logger> logger;
std::once_flag logger_init_flag;
std::once_flag file_init_flag;


std::mutex cout_mutex;

std::unordered_set<uintptr_t> testS;
std::mutex SET;







typedef cudaError_t (*cudaStreamWaitEvent_t)(cudaStream_t, cudaEvent_t, unsigned int);
typedef cudaError_t (*cudaEventRecord_t)(cudaEvent_t, cudaStream_t);
typedef cudaError_t (*cudaEventQuery_t)(cudaEvent_t);
typedef cudaError_t (*cudaEventDestroy_t)(cudaEvent_t);



typedef ncclResult_t  (*ncclReduce_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t  (*ncclBroadcast_t)(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
typedef ncclResult_t (*ncclAllReduce_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclReduceScatter_t)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclAllGather_t)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*ncclSendRecv_t)(const void*, size_t, ncclDataType_t, int, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);

typedef ncclResult_t (*ncclSend_t)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);                    
typedef ncclResult_t (*ncclRecv_t)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);



static cudaStreamWaitEvent_t real_cudaStreamWaitEvent = nullptr;
static cudaEventRecord_t real_cudaEventRecord = nullptr;
static cudaEventQuery_t real_cudaEventQuery = nullptr;
static cudaEventDestroy_t  real_cudaEventDestroy = nullptr;

static ncclReduce_t real_ncclReduce = NULL;
static ncclBroadcast_t real_ncclBroadcast = NULL;
static ncclAllReduce_t real_ncclAllReduce = NULL;
static ncclReduceScatter_t real_ncclReduceScatter = NULL;
static ncclAllGather_t real_ncclAllGather = NULL;
static ncclSendRecv_t real_ncclSendRecv = NULL;
static ncclSend_t real_ncclSend = NULL;
static ncclRecv_t real_ncclRecv = NULL;

std::string time_str;

std::unordered_map<int, std::shared_ptr<spdlog::logger>> rank_loggers;

void file_init(){
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm* local_tm = std::localtime(&now_time_t);
    std::ostringstream oss;
    oss << std::put_time(local_tm, "%Y%m%d_%H%M%S");
    time_str = oss.str();
    std::string folder_name = "logs/" + time_str;

    fs::create_directories(folder_name);

}

void init_logger(int rank) {
    std::call_once(file_init_flag, file_init);
    spdlog::init_thread_pool(8191, 4); // 初始化 spdlog 线程池
    spdlog::set_pattern("%v");

    std::string log_file = "logs/"+time_str+"/cuda_rank_" + std::to_string(rank) + ".log";
    logger = spdlog::basic_logger_mt<spdlog::async_factory>("cuda_logger_" + std::to_string(rank), log_file);
    rank_loggers[rank] = logger;
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info); // 设置全局日志级别
}

void ensure_logger_initialized(char* rank) {
    int rk = std::atoi(rank);
    std::call_once(logger_init_flag, [&](){init_logger(rk); });
}

void print_nccl_info(const char* func_name,cudaStream_t stream,long long t) {
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
//    std::cout <<"RANK: "<< getenv("OMPI_COMM_WORLD_RANK")<<" stream "<< stream<<" time "<< t<< " verb " << func_name << std::endl;
    std::string stream_event = std::to_string(reinterpret_cast<std::uintptr_t>(stream));
    // std::cout <<"RANK: " << getenv("OMPI_COMM_WORLD_RANK") << " stream " <<stream_event <<" "<<" Fuction "<<func_name<<std::endl;
    logger->info("[{}] [Rank: {}] NCCL Function {} called in stream {}" ,now_us_count,getenv("OMPI_COMM_WORLD_RANK"), func_name, stream_event);
    // logger->info("Rank {}  stream {} time {} verb {}  ", getenv("OMPI_COMM_WORLD_RANK"),stream_event,t,func_name);
}

void print_event_info(const char* func_name, cudaEvent_t event) {
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    // std::string str_event = std::to_string(reinterpret_cast<std::uintptr_t>(event));
    // std::cout<<getenv("OMPI_COMM_WORLD_RANK")<<" "<<func_name<<" "<<uintptr_t(event)<<std::endl;
    logger->info("[{}] [Rank: {}] Function {} called with event {}",now_us_count,getenv("OMPI_COMM_WORLD_RANK"), func_name, uintptr_t(event));
    // std::cout <<"RANK: "<< getenv("OMPI_COMM_WORLD_RANK")<< " Function " << func_name << " called with event " << event << std::endl;
}

void print_hang_info(EventInfo* ev,long long now_us_count){
    logger->info("Warning: [{}] [Rank: {}] event {} has been ongoing for [{}] ms", now_us_count, getenv("OMPI_COMM_WORLD_RANK"), uintptr_t(ev->event), 1.0*(now_us_count - (ev->start_time))/1000);
    std::cout<<"Warning:["<<now_us_count<<"] [Rank: "<< getenv("OMPI_COMM_WORLD_RANK")<< "] event "<<uintptr_t(ev->event)<<" has been ongoing for ["<<1.0*(now_us_count - (ev->start_time))/1000<<"] ms"<<std::endl;
}

void watchdog_thread() {
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
	int count;
    while (watchdog_state) {
	    count = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        // 从destroy_queue中取出event到destroy_event_list中
        {
            std::unique_lock<std::mutex> destroy_queue_lock(destroy_event_queue_mutex);
            while (!destroy_event_queue.empty()) {
                auto event_pair = destroy_event_queue.front();
                destroy_event_queue.pop();
                destroy_queue_lock.unlock();

                {
                    std::lock_guard<std::mutex> destroy_lock(destroy_event_list_mutex);
                    uintptr_t cur_event = uintptr_t(event_pair.first);
                    if(destroy_event_list.find(cur_event) == destroy_event_list.end())
                        destroy_event_list[cur_event] = event_pair.second;
                }

                destroy_queue_lock.lock();
            }
        }

        {
            std::unique_lock<std::mutex> event_queue_lock(event_queue_mutex);
            while (!event_queue.empty()) {
                auto event = event_queue.front();
                event_queue.pop();
                event_queue_lock.unlock();

                {
                    std::lock_guard<std::mutex> event_lock(event_list_mutex);
                    event_list.push_back(event);
                }
                event_queue_lock.lock();
            }
        }

	    {
            std::lock_guard<std::mutex> destroy_lock(destroy_event_list_mutex);
        //	std::vector<std::unordered_map<uintptr_t, EventInfo*>::iterator> iterVector;
            std::lock_guard<std::mutex> event_lock(event_list_mutex);
            Hang_Event.clear();
            for (auto it = event_list.begin(); it != event_list.end();){
                if ((*it) == nullptr || (*it)->event == nullptr) {
                    // std::cout<<"start nullptr"<<std::endl;
                    delete (*it);
     		        it = event_list.erase(it);
       			    continue;
    		    }
                if(destroy_event_list.find((uintptr_t)((*it)->event)) != destroy_event_list.end() && (*it)->destroy == false){
                    (*it)->destroy = true;
                    (*it)->end_time = destroy_event_list[(uintptr_t)((*it)->event)];
                }
                it++;
            }
            destroy_event_list.clear();
	    }

        {
            std::lock_guard<std::mutex> event_lock(event_list_mutex);
            // std::cout<<"start delete"<<std::endl;

            for (auto it = event_list.begin(); it != event_list.end(); ) {
                if ((*it) == nullptr || (*it)->event == nullptr) {
                    // std::cout<<"start nullptr"<<std::endl;
                    delete (*it);
                    it = event_list.erase(it);
                    continue;
                }
                if((*it)->destroy == true) {
                    logger->info("[{}] [Rank: {}] event {} has completed, start_time: {} end_time: {} exist_time {} ms",(*it)->end_time,getenv("OMPI_COMM_WORLD_RANK"), (uintptr_t)((*it)->event), (*it)->start_time, (*it)->end_time, 1.0*(((*it)->end_time)-((*it)->start_time))/1000) ;
                    // std::cout<<"Rank "<<getenv("OMPI_COMM_WORLD_RANK")<<" event " << (*it)->event<<" start_time:"<<(*it)->start_time<<" end_time:"<<(*it)->end_time<<" time "<<1.0*(((*it)->end_time)-((*it)->start_time))/1000<<" ms"<<std::endl;
                    std::lock_guard<std::mutex> SET_lock(SET);
                    if(testS.find((uintptr_t)((*it)->event)) != testS.end())
                        testS.erase((uintptr_t)((*it)->event));
                    delete (*it);
                    it = event_list.erase(it);
                }
                else {
                    auto event_ptr = (*it)->event;
                    if (event_ptr == nullptr) {
                        std::cout << "Event pointer is nullptr, skipping." << std::endl;
                         ++it;
                        continue;
                    }
                    {
                        std::lock_guard<std::mutex> SET_lock(SET);
                        if(testS.find((uintptr_t)((*it)->event)) != testS.end()) {it++; continue;}
                    }
                    auto Result = cudaEventQuery((*it)->event);


                    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    if (Result != cudaSuccess && Result != cudaErrorNotReady) {
                        delete (*it);
                        it = event_list.erase(it);
                        continue;
                    }

                    if(Result == cudaSuccess) {
                        std::lock_guard<std::mutex> SET_lock(SET);
                        delete (*it);
                        it = event_list.erase(it);
                        continue;
                    }
                    if(++((*it)->Life_time) >= Hang_Limit && Hang_Event.find((uintptr_t)((*it)->event)) == Hang_Event.end()){
                        Hang_Event.insert((uintptr_t)((*it)->event));
                        auto now = std::chrono::system_clock::now();
                        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
                        long long now_us_count = now_us.count();
                        if(now_us_count-((*it)->start_time) >= Hang_Time * 1000)
                            print_hang_info((*it),now_us_count);
                    }
                    it++;
                }
            }
        }
        {
    //         std::lock_guard<std::mutex> cout_lock(cout_mutex);
    //         std::cout<<std::dec;
            // std::cout<<"event_list_num:"<< static_cast<int>(event_list.size())<<std::endl;
    // //               std::cout<<"destroy_list_num:"<< static_cast<int>(destroy_event_list.size())<<std::endl;
    //         std::cout<<"event_list_ghost_num:" << count<<std::endl;
    //         std::cout<<"all_dstroy_cnt:"<<destroy_count<<std::endl;
        }
	    if (event_list.empty()) {

       		std::lock_guard<std::mutex> guard(running_mutex);
       		watchdog_state.exchange(false);
		    break;
    	}

//	    std::cout<<"release lock"<<std::endl;
    }
	// std::cout<<"exit thread"<<std::endl;
}
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!real_cudaStreamWaitEvent) {
        real_cudaStreamWaitEvent = (cudaStreamWaitEvent_t)dlsym(RTLD_NEXT, "cudaStreamWaitEvent");
    }
    auto now = std::chrono::system_clock::now();
	auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
  	long long now_us_count = now_us.count();
    print_event_info("cudaStreamWaitEvent", event);
    std::lock_guard<std::mutex> SET_lock(SET);
    if(testS.find((uintptr_t)event) != testS.end()){}
    else{
//	std::cout<<"Insert Event"<<std::endl;
    	EventInfo* ev_info = new EventInfo(event,stream,now_us_count);
        ev_info->Life_time = 0;
	    {
		    std::lock_guard<std::mutex> event_queue_lock(event_queue_mutex);
    		event_queue.push(ev_info);
    	}
   	    if (!watchdog_state.exchange(true)) {
		    // std::cout<<"create thread"<<std::endl;
        	std::thread(watchdog_thread).detach();
    	}
   }
   return real_cudaStreamWaitEvent(stream, event, flags);
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!real_cudaEventRecord) {
        real_cudaEventRecord = (cudaEventRecord_t)dlsym(RTLD_NEXT, "cudaEventRecord");
    }

    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));

    print_event_info("cudaEventRecord", event);
    return real_cudaEventRecord(event,stream);
}
// 拦截 cudaEventQuery
extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!real_cudaEventQuery) {
        real_cudaEventQuery = (cudaEventQuery_t)dlsym(RTLD_NEXT, "cudaEventQuery");
    }
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    // print_event_info("cudaEventQuery", event);
    return real_cudaEventQuery(event);
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (!real_cudaEventDestroy) {
        real_cudaEventDestroy = (cudaEventDestroy_t)dlsym(RTLD_NEXT, "cudaEventDestroy");
    }
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    auto now = std::chrono::system_clock::now();
	auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
  	long long now_us_count = now_us.count();

    std::lock_guard<std::mutex> SET_lock(SET);
    if(testS.find((uintptr_t)event) != testS.end()){}
    else{
	    testS.insert((uintptr_t)event);
        {
            std::lock_guard<std::mutex> destroy_queue_lock(destroy_event_queue_mutex);
            destroy_event_queue.push(std::make_pair(event, now_us_count));
        }

	    // std::lock_guard<std::mutex> destroy_lock(destroy_event_list_mutex);
	    // if(destroy_event_list.find(uintptr_t(event)) == destroy_event_list.end()){
	   	//     destroy_event_list.insert(std::make_pair(uintptr_t(event),now_us_count));
	    // }

    }
    //print_event_info("cudaEventDestroy",event);
    return real_cudaEventDestroy(event);
}


extern "C" ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclAllReduce_t real_ncclAllReduce = (ncclAllReduce_t)dlsym(handle, "ncclAllReduce");
    if (!real_ncclAllReduce) {
        fprintf(stderr, "Cannot find symbol ncclAllReduce: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclAllReduce",stream,now_us_count);
    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

extern "C" ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclReduceScatter_t real_ncclReduceScatter = (ncclReduceScatter_t)dlsym(handle, "ncclReduceScatter");
    if (!real_ncclReduceScatter) {
        fprintf(stderr, "Cannot find symbol ncclReduceScatter: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclReduceScatter",stream,now_us_count);
    return real_ncclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

extern "C" ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclAllGather_t real_ncclAllGather = (ncclAllGather_t)dlsym(handle, "ncclAllGather");
    if (!real_ncclAllGather) {
        fprintf(stderr, "Cannot find symbol ncclAllGather: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclAllGather",stream,now_us_count);
    return real_ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
}

extern "C" ncclResult_t ncclSendRecv(const void* sendbuff, size_t sendcount, ncclDataType_t sendtype, int peer_send, void* recvbuff, size_t recvcount, ncclDataType_t recvtype, int peer_recv, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclSendRecv_t real_ncclSendRecv = (ncclSendRecv_t)dlsym(handle, "ncclSendRecv");
    if (!real_ncclSendRecv) {
        fprintf(stderr, "Cannot find symbol ncclSendRecv: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclSendRecv",stream,now_us_count);
    return real_ncclSendRecv(sendbuff, sendcount, sendtype, peer_send, recvbuff, recvcount, recvtype, peer_recv, comm, stream);
}


extern "C" ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclSend_t real_ncclSend = (ncclSend_t)dlsym(handle, "ncclSend");
    if (!real_ncclSend) {
        fprintf(stderr, "Cannot find symbol ncclSend: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclSend",stream,now_us_count);
    return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);
}
extern "C" ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclRecv_t real_ncclRecv = (ncclRecv_t)dlsym(handle, "ncclRecv");
    if (!real_ncclRecv) {
        fprintf(stderr, "Cannot find symbol ncclSendRecv: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclRecv",stream,now_us_count);
    return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);
}


extern "C" ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclReduce_t real_ncclReduce = (ncclReduce_t)dlsym(handle, "ncclReduce");
    if (!real_ncclReduce) {
        fprintf(stderr, "Cannot find symbol ncclReduce: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclReduce",stream,now_us_count);
    return real_ncclReduce(sendbuff, recvbuff, count,  datatype, op, root,  comm, stream);
}

extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    void* handle = dlopen("libnccl.so", RTLD_LAZY);
    ensure_logger_initialized(getenv("OMPI_COMM_WORLD_RANK"));
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
    ncclBroadcast_t real_ncclBroadcast = (ncclBroadcast_t)dlsym(handle, "ncclBroadcast");
    if (!real_ncclBroadcast) {
        fprintf(stderr, "Cannot find symbol ncclBroadcast: %s\n", dlerror());
        dlclose(handle);
    }
    dlclose(handle);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    long long now_us_count = now_us.count();
    print_nccl_info("ncclBroadcast",stream,now_us_count);
    return real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}




