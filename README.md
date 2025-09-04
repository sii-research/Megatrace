# Megatrace NCCL Interceptor

A high-performance interception library for NCCL (NVIDIA Collective Communications Library) functions with comprehensive logging and tracing capabilities.

## Features

- **NCCL Function Interception**: Intercepts all major NCCL collective communication functions
- **High Performance**: Lock-free thread initialization with minimal overhead
- **Ring Buffer Logging**: Efficient asynchronous logging with ring buffer
- **Automatic Initialization**: Self-initializing with constructor attributes

## Getting Started

### Build

```bash
# Set up CUDA environment
vim ~/.bashrc 
export CPATH=/usr/local/cuda/include:$CPATH

# Compile the interceptor library
g++ -shared -o nccl_intercept.so nccl_intercept_main.cpp ring_log.cc -ldl -fPIC -lpthread
```

### Usage

```bash
# Basic usage
LD_PRELOAD=/path/to/megatrace_intercept/nccl_intercept.so your_mpi_program

# With custom log path
MEGATRACE_PATH=/path/to/logs/ LD_PRELOAD=/path/to/megatrace_intercept/nccl_intercept.so your_mpi_program
```

## Logging System

### Log Levels

The interceptor supports four log levels controlled by the `MEGATRACE_LOG_LEVEL` environment variable:

- **ERROR** (default): Only error messages
- **WARN**: Warning and error messages  
- **INFO**: Information, warning, and error messages
- **DEBUG**: All messages including debug information

### Environment Variables

```bash
# Set log level (default: ERROR)
export MEGATRACE_LOG_LEVEL=INFO

# Set log output path
export MEGATRACE_PATH=/path/to/logs/

# MPI rank information (automatically detected)
export OMPI_COMM_WORLD_RANK=0
```

### Log Format

```
[2024-01-15 14:30:25.123456] [Rank: 0] [ERROR] [nccl_intercept_main.cpp:82:ncclAllReduce] Cannot find symbol ncclAllReduce: undefined symbol
```

Format: `[timestamp] [Rank: X] [LEVEL] [file:line:function] message`

### Example Usage

```bash
# Run with INFO level logging
MEGATRACE_LOG_LEVEL=INFO LD_PRELOAD=./nccl_intercept.so mpirun -np 4 your_program

# Run with custom log directory
MEGATRACE_PATH=/tmp/megatrace_logs/ MEGATRACE_LOG_LEVEL=WARN LD_PRELOAD=./nccl_intercept.so your_program
```

## Intercepted Functions

The library intercepts the following NCCL functions:

- `ncclAllReduce` - All-reduce collective operation
- `ncclReduceScatter` - Reduce-scatter collective operation  
- `ncclAllGather` - All-gather collective operation
- `ncclSendRecv` - Send-receive operation
- `ncclSend` - Send operation
- `ncclRecv` - Receive operation
- `ncclReduce` - Reduce collective operation
- `ncclBroadcast` - Broadcast collective operation

## Performance Considerations

- **Minimal Overhead**: Uses atomic operations and branch prediction hints for optimal performance
- **Lock-free Initialization**: Thread initialization uses compare-and-swap operations
- **Efficient Logging**: Ring buffer with background thread for non-blocking logging
- **Default Error-only**: By default only logs errors to minimize performance impact

## Architecture

- **Dynamic Loading**: Uses `dlsym` with `RTLD_NEXT` for function interception
- **Thread Management**: Background logging thread with atomic state management
- **Memory Management**: Ring buffer for efficient log storage
- **Error Handling**: Comprehensive error checking and reporting

