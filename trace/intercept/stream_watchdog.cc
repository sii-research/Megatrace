#include "stream_watchdog.h"

#ifdef MEGATRACE_GPU_NVIDIA

#include "log.h"

#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <chrono>

// Watchdog configuration (can be tuned later or made env-driven)
static const int kWatchdogSleepMs = 1000;      // interval between scans
static const int kHangLimitIters = 5;          // number of iterations before considering hang
static const long long kHangTimeMs = 5000;     // minimum lifetime in ms before reporting hang

// Queues for producer/consumer pattern
static std::queue<StreamEventInfo*> g_event_queue;
static std::mutex g_event_queue_mutex;

// Optional destroy queue if we later intercept cudaEventDestroy
static std::queue<std::pair<cudaEvent_t, long long>> g_destroy_event_queue;
static std::mutex g_destroy_event_queue_mutex;

// Active event list
static std::vector<StreamEventInfo*> g_active_events;
static std::mutex g_active_events_mutex;

// Map from event pointer to destroy time (microseconds)
static std::unordered_map<uintptr_t, long long> g_destroy_event_map;
static std::mutex g_destroy_event_map_mutex;

// Track events we've already reported as hung to avoid spamming logs
static std::unordered_set<uintptr_t> g_reported_hang_events;
static std::mutex g_reported_hang_events_mutex;

// Thread state
static std::atomic<bool> g_stream_watchdog_running(false);
static std::mutex g_watchdog_state_mutex;

static long long now_us() {
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return now_us.count();
}

static void stream_watchdog_thread() {
    while (g_stream_watchdog_running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kWatchdogSleepMs));

        // Drain destroy_event_queue into map
        {
            std::unique_lock<std::mutex> lock(g_destroy_event_queue_mutex);
            while (!g_destroy_event_queue.empty()) {
                auto ev = g_destroy_event_queue.front();
                g_destroy_event_queue.pop();
                lock.unlock();

                {
                    std::lock_guard<std::mutex> map_lock(g_destroy_event_map_mutex);
                    uintptr_t key = reinterpret_cast<uintptr_t>(ev.first);
                    // record destroy time if not already present
                    if (g_destroy_event_map.find(key) == g_destroy_event_map.end()) {
                        g_destroy_event_map[key] = ev.second;
                    }
                }

                lock.lock();
            }
        }

        // Drain event_queue into active_events
        {
            std::unique_lock<std::mutex> qlock(g_event_queue_mutex);
            while (!g_event_queue.empty()) {
                StreamEventInfo* info = g_event_queue.front();
                g_event_queue.pop();
                qlock.unlock();

                if (info && info->event) {
                    std::lock_guard<std::mutex> alock(g_active_events_mutex);
                    g_active_events.push_back(info);
                } else if (info) {
                    delete info;
                }

                qlock.lock();
            }
        }

        // Apply destroy times to active events, then clear map
        {
            std::lock_guard<std::mutex> map_lock(g_destroy_event_map_mutex);
            std::lock_guard<std::mutex> alock(g_active_events_mutex);
            for (auto* info : g_active_events) {
                if (!info || !info->event) continue;
                uintptr_t key = reinterpret_cast<uintptr_t>(info->event);
                auto it = g_destroy_event_map.find(key);
                if (it != g_destroy_event_map.end() && !info->destroyed) {
                    info->destroyed = true;
                    info->end_time_us = it->second;
                }
            }
            g_destroy_event_map.clear();
        }

        // Scan active events
        {
            std::lock_guard<std::mutex> alock(g_active_events_mutex);
            for (auto it = g_active_events.begin(); it != g_active_events.end();) {
                StreamEventInfo* info = *it;
                if (info == nullptr || info->event == nullptr) {
                    delete info;
                    it = g_active_events.erase(it);
                    continue;
                }

                cudaError_t result = cudaEventQuery(info->event);
                if (result != cudaSuccess && result != cudaErrorNotReady) {
                    // Invalid state, drop it
                    delete info;
                    it = g_active_events.erase(it);
                    continue;
                }

                if (result == cudaSuccess) {
                    // Completed; if destroy time was recorded, we can log lifetime if desired
                    delete info;
                    it = g_active_events.erase(it);
                    continue;
                }

                // Still not ready
                info->life_time += 1;
                long long now_ts = now_us();
                long long lived_ms = (now_ts - info->start_time_us) / 1000;

                if (info->life_time >= kHangLimitIters && lived_ms >= kHangTimeMs) {
                    uintptr_t key = reinterpret_cast<uintptr_t>(info->event);
                    bool should_report = false;
                    {
                        std::lock_guard<std::mutex> rlock(g_reported_hang_events_mutex);
                        if (g_reported_hang_events.find(key) == g_reported_hang_events.end()) {
                            g_reported_hang_events.insert(key);
                            should_report = true;
                        }
                    }
                    if (should_report) {
                        const char* rank = getenv("OMPI_COMM_WORLD_RANK");
                        if (!rank) rank = getenv("RANK");
                        if (!rank) rank = "0";
                        LOG_WARN("Watchdog: cudaEvent %p on stream %p potentially hung; rank=%s, "
                                 "start_us=%lld, now_us=%lld, lifetime_ms=%lld",
                                 static_cast<void*>(info->event),
                                 static_cast<void*>(info->stream),
                                 rank,
                                 (long long)info->start_time_us,
                                 (long long)now_ts,
                                 (long long)lived_ms);
                    }
                }

                ++it;
            }
        }

        // If no active events remain, stop the watchdog
        {
            std::lock_guard<std::mutex> alock(g_active_events_mutex);
            if (g_active_events.empty()) {
                std::lock_guard<std::mutex> state_lock(g_watchdog_state_mutex);
                g_stream_watchdog_running.store(false, std::memory_order_release);
                break;
            }
        }
    }
}

void enqueue_stream_event(cudaEvent_t event, gpu_stream_t stream, long long start_time_us) {
    if (event == nullptr) {
        return;
    }

    // Allocate event info
    StreamEventInfo* info = new StreamEventInfo(event, stream, start_time_us);

    // Push into queue
    {
        std::lock_guard<std::mutex> lock(g_event_queue_mutex);
        g_event_queue.push(info);
    }

    // Lazily start watchdog thread
    bool expected = false;
    if (g_stream_watchdog_running.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::thread(stream_watchdog_thread).detach();
    }
}

#endif // MEGATRACE_GPU_NVIDIA


