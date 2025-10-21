#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdarg.h>

// Log levels
typedef enum {
    LOG_ERROR = 0,
    LOG_WARN  = 1,
    LOG_INFO  = 2,
    LOG_DEBUG = 3
} log_level_t;

// Global log level (default to ERROR only)
static log_level_t g_log_level = LOG_ERROR;

// Initialize log system - call this once at startup
static inline void log_init() {
    const char* env_level = getenv("MEGATRACE_LOG_LEVEL");
    if (env_level != NULL) {
        if (strcmp(env_level, "ERROR") == 0 || strcmp(env_level, "error") == 0) {
            g_log_level = LOG_ERROR;
        } else if (strcmp(env_level, "WARN") == 0 || strcmp(env_level, "warn") == 0) {
            g_log_level = LOG_WARN;
        } else if (strcmp(env_level, "INFO") == 0 || strcmp(env_level, "info") == 0) {
            g_log_level = LOG_INFO;
        } else if (strcmp(env_level, "DEBUG") == 0 || strcmp(env_level, "debug") == 0) {
            g_log_level = LOG_DEBUG;
        } else {
            // Invalid level, keep default
            g_log_level = LOG_ERROR;
        }
    }
}

// Get current timestamp string
static inline void get_timestamp(char* buffer, size_t size) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    
    struct tm* tm_info = localtime(&tv.tv_sec);
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Append microseconds
    char microsec[8];
    snprintf(microsec, sizeof(microsec), ".%06ld", tv.tv_usec);
    strncat(buffer, microsec, size - strlen(buffer) - 1);
}

// Get rank from environment
static inline const char* get_rank_str() {
    const char* rank = getenv("OMPI_COMM_WORLD_RANK");
    return rank ? rank : "0";
}

// Core logging function
static inline void log_print(log_level_t level, const char* level_str, const char* file, int line, const char* func, const char* format, ...) {
    if (level > g_log_level) {
        return; // Skip if level is too high
    }
    
    char timestamp[32];
    get_timestamp(timestamp, sizeof(timestamp));
    
    const char* rank = get_rank_str();
    
    // Print header: [timestamp] [Rank: X] [LEVEL] [file:line:func]
    fprintf(stderr, "[%s] [Rank: %s] [%s] [%s:%d:%s] ", 
            timestamp, rank, level_str, file, line, func);
    
    // Print message
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    fflush(stderr);
}

// Logging macros
#define LOG_ERROR(fmt, ...) log_print(LOG_ERROR, "ERROR", __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_print(LOG_WARN,  "WARN",  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  log_print(LOG_INFO,  "INFO",  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) log_print(LOG_DEBUG, "DEBUG", __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

// Convenience macros for common cases
#define LOG_ERROR_SIMPLE(fmt, ...) log_print(LOG_ERROR, "ERROR", __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN_SIMPLE(fmt, ...)  log_print(LOG_WARN,  "WARN",  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_INFO_SIMPLE(fmt, ...)  log_print(LOG_INFO,  "INFO",  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

// Check if logging is enabled for a level
#define LOG_IS_ERROR_ENABLED() (g_log_level >= LOG_ERROR)
#define LOG_IS_WARN_ENABLED()  (g_log_level >= LOG_WARN)
#define LOG_IS_INFO_ENABLED()  (g_log_level >= LOG_INFO)
#define LOG_IS_DEBUG_ENABLED() (g_log_level >= LOG_DEBUG)

#endif // LOG_H
