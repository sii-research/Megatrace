# Megatrace 硬编码值清单

本文档列出了代码中所有发现的硬编码数字、固定值和环境变量依赖，供配置化参考。

## C/C++ 拦截器代码

### ring_log.h

| 行号 | 配置项 | 当前值 | 类型 | 说明 |
|------|--------|--------|------|------|
| 17 | `RING_BUFFER_SIZE` | 10000 | 宏定义 | 环形缓冲区容量 |
| 18 | `LOG_MAX_LEN` | 256 | 宏定义 | 日志条目最大长度（字节） |
| 19 | `BATCH_SIZE` | 10240 | 宏定义 | 批处理大小 |
| 20 | `FLUSH_INTERVAL_MS` | 4000 | 宏定义 | 刷新间隔（毫秒） |
| 22 | `LOG_ROTATE_SIZE` | 1024 * 1024 | 宏定义 | 日志轮转大小（1MB） |
| 23 | `MAX_LOG_VERSIONS` | 3 | 宏定义 | 最大版本数 |

### nccl_intercept.cc

| 行号 | 配置项 | 当前值 | 类型 | 说明 |
|------|--------|--------|------|------|
| 162 | `usleep(10000)` | 10000 | 硬编码 | 线程启动等待时间（微秒，10ms） |
| 172 | `for (i < 1000)` | 1000 | 硬编码 | 自旋等待循环次数 |
| 201-211 | 库名称列表 | 硬编码数组 | 硬编码 | PyTorch/NCCL库搜索列表 |

### ring_log.cc

| 行号 | 配置项 | 当前值 | 类型 | 说明 |
|------|--------|--------|------|------|
| 21 | `nccl_megatrace_enable` | 1 | 环境变量默认值 | 启用标志 |
| 22 | `nccl_megatrace_log_path` | "./logs" | 环境变量默认值 | 日志路径 |
| 23 | `nccl_sensitive_time` | 3000 | 环境变量默认值 | 灵敏度时间（毫秒） |
| 100 | `dash_count < 8` | 8 | 硬编码 | Pod名称解析阈值 |
| 116-117 | `old_name[512]`, `new_name[512]` | 512 | 硬编码 | 文件名缓冲区大小 |
| 189 | `hostname[256]` | 256 | 硬编码 | 主机名缓冲区大小 |
| 207 | `time_buffer[80]` | 80 | 硬编码 | 时间字符串缓冲区大小 |
| 212 | `filename[256]` | 256 | 硬编码 | 文件名缓冲区大小 |
| 273 | `save_iter % 300` | 300 | 硬编码 | futimens调用间隔 |
| 278 | `sleep(1)` | 1 | 硬编码 | 日志写入线程睡眠时间（秒） |
| 287 | `time_str[64]` | 64 | 硬编码 | 时间戳字符串缓冲区大小 |
| 295 | `pci_bus_id[32]` | 32 | 硬编码 | PCI总线ID缓冲区大小 |

### log.h

| 行号 | 配置项 | 当前值 | 类型 | 说明 |
|------|--------|--------|------|------|
| 25 | `MEGATRACE_LOG_LEVEL` | LOG_ERROR | 环境变量 | 日志级别默认值 |
| 51 | `microsec[8]` | 8 | 硬编码 | 微秒字符串缓冲区大小 |
| 59-61 | `OMPI_COMM_WORLD_RANK` / `RANK` | "0" | 环境变量 | Rank编号默认值 |
| 76 | `timestamp[32]` | 32 | 硬编码 | 时间戳缓冲区大小 |

### 环境变量依赖

| 环境变量 | 文件 | 行号 | 默认值 | 说明 |
|----------|------|------|--------|------|
| `NCCL_MEGATRACE_ENABLE` | ring_log.cc | 21 | 1 | 启用/禁用日志 |
| `NCCL_MEGATRACE_LOG_PATH` | ring_log.cc | 22 | "./logs" | 日志文件路径 |
| `NCCL_MEGATRACE_SENSTIME` | ring_log.cc | 23 | 3000 | 批量刷写灵敏度（毫秒） |
| `POD` | ring_log.cc | 177 | "unknown" | Pod名称 |
| `MY_POD_IP` | ring_log.cc | 183 | "unknown" | Pod IP地址 |
| `TRAIN_JOB_ID` | ring_log.cc | 194 | "unknown" | 训练任务ID |
| `OMPI_COMM_WORLD_RANK` | log.h | 59 | "0" | Rank编号（MPI） |
| `RANK` | log.h | 61 | "0" | Rank编号（torchrun） |
| `MEGATRACE_LOG_LEVEL` | log.h | 25 | LOG_ERROR | 日志级别 |

## Python 分析代码

### trace/analysis/config.py

| 配置项路径 | 当前值 | 类型 | 说明 |
|-----------|--------|------|------|
| `HANG_ANALYSIS_CONFIG.timeout_threshold` | 300 | 硬编码 | 超时阈值（秒） |
| `HANG_ANALYSIS_CONFIG.operation_interval_threshold` | 60 | 硬编码 | 操作间隔阈值（秒） |
| `HANG_ANALYSIS_CONFIG.collective_hang_threshold` | 120 | 硬编码 | 集合操作hang阈值（秒） |
| `HANG_ANALYSIS_CONFIG.send_recv_hang_threshold` | 180 | 硬编码 | 发送接收hang阈值（秒） |
| `HANG_ANALYSIS_CONFIG.stream_inactive_threshold` | 300 | 硬编码 | 流无活动阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.slow_threshold` | 10 | 硬编码 | 慢操作阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.operation_performance_thresholds.AllReduce` | 5 | 硬编码 | AllReduce性能阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.operation_performance_thresholds.AllGather` | 8 | 硬编码 | AllGather性能阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.operation_performance_thresholds.Broadcast` | 3 | 硬编码 | Broadcast性能阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.operation_performance_thresholds.Send` | 15 | 硬编码 | Send性能阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.operation_performance_thresholds.Recv` | 15 | 硬编码 | Recv性能阈值（秒） |
| `SLOW_ANALYSIS_CONFIG.data_size_thresholds.small` | 1024 | 硬编码 | 小数据阈值（字节） |
| `SLOW_ANALYSIS_CONFIG.data_size_thresholds.medium` | 1048576 | 硬编码 | 中等数据阈值（字节，1MB） |
| `SLOW_ANALYSIS_CONFIG.data_size_thresholds.large` | 10485760 | 硬编码 | 大数据阈值（字节，10MB） |
| `SLOW_ANALYSIS_CONFIG.data_size_thresholds.huge` | 104857600 | 硬编码 | 超大数据阈值（字节，100MB） |
| `ANALYSIS_DEPTH_CONFIG.max_file_size_mb` | 1000 | 硬编码 | 最大文件大小（MB） |
| `ANALYSIS_DEPTH_CONFIG.max_lines` | 1000000 | 硬编码 | 最大行数 |
| `ANALYSIS_DEPTH_CONFIG.thread_pool_size` | 4 | 硬编码 | 线程池大小 |

### trace/analysis/parallel_slow_detector.py

| 配置项 | 当前值 | 类型 | 说明 |
|--------|--------|------|------|
| `GRUBBS_CRITICAL_VALUES` | 硬编码字典 | 硬编码 | Grubbs测试临界值（n=4到30） |

### trace/analysis/group_hash_slow_detector.py

| 配置项 | 当前值 | 类型 | 说明 |
|--------|--------|------|------|
| `GRUBBS_CRITICAL_VALUES` | 硬编码字典 | 硬编码 | Grubbs测试临界值（n=4到30） |

## 告警系统代码

### bot/alert-to-feishu/config.py

| 配置项 | 当前值 | 类型 | 说明 |
|--------|--------|------|------|
| `ALERTMANAGER_TIMEOUT` | 10 | 硬编码 | AlertManager超时（秒） |
| `ALERTMANAGER_MAX_RETRIES` | 3 | 硬编码 | 最大重试次数 |
| `ALERTMANAGER_POLL_INTERVAL` | 60 | 硬编码 | 轮询间隔（秒） |
| `RECURRENCE_INTERVAL_SECONDS` | 1200 | 硬编码 | 复发告警间隔（秒，20分钟） |
| `REPEAT_NOTIFICATION_INTERVAL_SECONDS` | 86400 | 硬编码 | 重复推送间隔（秒，24小时） |
| `DATABASE_BACKUP_DAYS` | 7 | 硬编码 | 数据库备份保留天数 |
| `LOG_MAX_SIZE` | 100 | 硬编码 | 日志文件最大大小（MB） |
| `LOG_BACKUP_COUNT` | 5 | 硬编码 | 日志备份数量 |
| `MAX_MEMORY_MB` | 1024 | 硬编码 | 最大内存使用量（MB） |
| `MAX_CPU_PERCENT` | 90 | 硬编码 | 最大CPU使用率（%） |

### 环境变量支持

| 环境变量 | 配置项 | 说明 |
|----------|--------|------|
| `ALERTMANAGER_URL` | `ALERTMANAGER_URL` | AlertManager地址 |
| `ALERTMANAGER_TIMEOUT` | `ALERTMANAGER_TIMEOUT` | AlertManager超时 |
| `DATABASE_PATH` | `DATABASE_PATH` | 数据库路径 |
| `LOG_FILE` | `LOG_FILE` | 日志文件路径 |
| `LOG_LEVEL` | `LOG_LEVEL` | 日志级别 |

## 配置化优先级

### 高优先级（立即配置化）

1. ✅ **性能相关参数**
   - 环形缓冲区大小
   - 批处理大小
   - 刷新间隔

2. ✅ **路径和目录**
   - 日志文件路径
   - 数据库路径
   - 输出目录

3. ✅ **阈值参数**
   - Hang检测阈值
   - 慢操作阈值
   - 性能阈值

### 中优先级（近期配置化）

1. ⚠️ **库搜索列表**
   - PyTorch/NCCL库名称列表

2. ⚠️ **缓冲区大小**
   - 各种字符串缓冲区大小

3. ⚠️ **时间间隔**
   - 线程睡眠间隔
   - 轮询间隔

### 低优先级（可选配置化）

1. 📝 **统计值**
   - Grubbs临界值（统计常数，通常不变）

2. 📝 **固定常量**
   - 一些算法相关的固定值

## 统计信息

- **C/C++硬编码值总数**: ~20个
- **Python硬编码值总数**: ~25个
- **环境变量依赖总数**: ~12个
- **需要配置化的总数**: ~45个

## 相关文档

- [配置化方案详细文档](../CONFIGURATION_PLAN.md)
- [配置使用指南](./README.md)
- [配置文件示例](./intercept_config.yaml.example)
- [配置文件示例](./analysis_config.yaml.example)
- [配置文件示例](./alert_config.yaml.example)

---

**最后更新**: 2024-01-XX  
**维护者**: Megatrace Team

