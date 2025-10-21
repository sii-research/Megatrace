# Megatrace

一个用于拦截 NCCL（NVIDIA Collective Communications Library）主要函数并进行高性能日志与追踪的动态库。

## 特性

- **NCCL 函数拦截**：拦截主要的 NCCL 集体通信与点对点函数
- **高性能**：使用原子与分支预测，初始化与日志开销极低
- **环形缓冲日志**：后台线程异步批量刷写，避免阻塞主流程
- **自动初始化**：使用构造器属性自动完成初始化

## 开始使用

### 编译

```bash
# 配置 CUDA 环境
vim ~/.bashrc
export CPATH=/usr/local/cuda/include:$CPATH

# 编译拦截库
g++ -shared -o nccl_intercept.so nccl_intercept_main.cpp ring_log.cc -ldl -fPIC -lpthread
```

### 运行

```bash
# 基本用法
LD_PRELOAD=/path/to/megatrace_intercept/nccl_intercept.so your_mpi_program

# 启用并自定义日志目录
NCCL_MEGATRACE_ENABLE=1 NCCL_MEGATRACE_LOG_PATH=/path/to/logs/ \
  LD_PRELOAD=/path/to/megatrace_intercept/nccl_intercept.so your_mpi_program
```

## 日志系统

### 日志级别

通过环境变量 `MEGATRACE_LOG_LEVEL` 控制输出到标准错误的日志级别：

- **ERROR**（默认）：仅错误
- **WARN**：警告与错误
- **INFO**：信息、警告与错误
- **DEBUG**：包含调试在内的全部信息

### 环境变量

```bash
# 启用/关闭 Megatrace 文件日志（默认：1）
# 1：启用；0：关闭
export NCCL_MEGATRACE_ENABLE=1

# 每个 rank 的日志文件存放目录（启用时生效；默认：./logs）
export NCCL_MEGATRACE_LOG_PATH=/path/to/logs/

# 低流量场景下的批量刷写灵敏度，毫秒（默认：3000）
export NCCL_MEGATRACE_SENSTIME=3000

# 控制控制台（stderr）的日志级别（默认：ERROR）
# 可选：ERROR、WARN、INFO、DEBUG
export MEGATRACE_LOG_LEVEL=INFO

# MPI rank（通常由 MPI 运行时自动设置，此处仅示例）
export OMPI_COMM_WORLD_RANK=0
```

说明：
- `NCCL_MEGATRACE_ENABLE` 控制是否将拦截到的事件写入环形缓冲并最终刷入文件。
- 当 `NCCL_MEGATRACE_ENABLE` 启用时，日志文件会写入到 `NCCL_MEGATRACE_LOG_PATH`（默认 `./logs`）。
- `NCCL_MEGATRACE_SENSTIME` 用于调节后台线程在低日志量时的刷写间隔敏感度。
- `MEGATRACE_LOG_LEVEL` 仅影响拦截器本身输出到标准错误的控制台日志，不影响文件日志内容。

### 日志格式

```
[2024-01-15 14:30:25.123456] [Rank: 0] [ERROR] [nccl_intercept_main.cpp:82:ncclAllReduce] Cannot find symbol ncclAllReduce: undefined symbol
```

格式：`[timestamp] [Rank: X] [LEVEL] [file:line:function] message`

### 使用示例

```bash
# 以 INFO 级别运行
MEGATRACE_LOG_LEVEL=INFO LD_PRELOAD=./nccl_intercept.so mpirun -np 4 your_program

# 启用文件日志并自定义目录
NCCL_MEGATRACE_ENABLE=1 NCCL_MEGATRACE_LOG_PATH=/tmp/megatrace_logs/ \
  MEGATRACE_LOG_LEVEL=WARN LD_PRELOAD=./nccl_intercept.so your_program
```

## 拦截的函数

拦截库会拦截以下 NCCL 函数：

- `ncclAllReduce` - 全规约
- `ncclReduceScatter` - 规约散射
- `ncclAllGather` - 全收集
- `ncclSendRecv` - 发送接收
- `ncclSend` - 发送
- `ncclRecv` - 接收
- `ncclReduce` - 规约
- `ncclBroadcast` - 广播

## 性能注意事项

- **极低开销**：使用原子操作与分支预测降低拦截与日志的性能影响
- **无锁初始化**：初始化流程使用 CAS 原子策略
- **高效日志**：环形缓冲与后台线程避免阻塞主线程
- **默认仅错误**：默认仅输出错误日志，尽可能降低开销

## 架构概览

- **动态链接**：通过 `dlsym` 实现函数拦截
- **线程管理**：后台日志线程使用原子状态管理
- **内存管理**：环形缓冲存储日志
- **错误处理**：完整的错误检查与报告
