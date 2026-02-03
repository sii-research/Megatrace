#ifndef STREAM_WATCHDOG_H
#define STREAM_WATCHDOG_H

#include "gpu_config.h"

#ifdef MEGATRACE_GPU_NVIDIA

#include <cuda_runtime.h>
#include <atomic>

// Lightweight event info for watchdog tracking
struct StreamEventInfo {
    cudaEvent_t event;
    gpu_stream_t stream;
    long long start_time_us;
    long long end_time_us;
    int life_time;      // number of watchdog iterations seen
    bool destroyed;     // whether a destroy time has been observed

    StreamEventInfo(cudaEvent_t ev, gpu_stream_t s, long long start_us)
        : event(ev),
          stream(s),
          start_time_us(start_us),
          end_time_us(0),
          life_time(0),
          destroyed(false) {}
};

// Enqueue a stream event for watchdog monitoring.
// Safe to call from interceptors; starts watchdog thread lazily.
void enqueue_stream_event(cudaEvent_t event, gpu_stream_t stream, long long start_time_us);

#endif // MEGATRACE_GPU_NVIDIA

#endif // STREAM_WATCHDOG_H


