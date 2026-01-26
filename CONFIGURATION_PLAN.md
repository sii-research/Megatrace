# Megatrace 配置化方案

本文档整理了代码中所有硬编码的数字、固定的环境变量依赖等可配置部分，并提供了统一的配置化方案。

## 一、C/C++ 拦截器代码中的硬编码值

### 1.1 ring_log.h 中的宏定义

| 配置项 | 当前值 | 说明 | 建议配置化 |
|--------|--------|------|-----------|
| `RING_BUFFER_SIZE` | 10000 | 环形缓冲区容量 | ✅ 是 |
| `LOG_MAX_LEN` | 256 | 日志条目最大长度 | ✅ 是 |
| `BATCH_SIZE` | 10240 | 工作线程每批处理的条目数 | ✅ 是 |
| `FLUSH_INTERVAL_MS` | 4000 | 定期刷新间隔（毫秒） | ✅ 是 |
| `LOG_ROTATE_SIZE` | 1024 * 1024 (1MB) | 日志文件轮转大小 | ✅ 是 |
| `MAX_LOG_VERSIONS` | 3 | 保留的轮转日志文件数量 | ✅ 是 |
| `MEGATRACE_LOG_ENABLE` | 1 | 日志启用标志 | ✅ 是（已有环境变量） |

### 1.2 nccl_intercept.cc 中的硬编码值

| 配置项 | 当前值 | 位置 | 说明 | 建议配置化 |
|--------|--------|------|------|-----------|
| `usleep(10000)` | 10000 | 第162行 | 线程启动等待时间（10ms） | ✅ 是 |
| `for (volatile int i = 0; i < 1000; i++)` | 1000 | 第172行 | 自旋等待循环次数 | ✅ 是 |
| 库名称列表 | 硬编码数组 | 第201-211行 | PyTorch/NCCL库搜索列表 | ✅ 是 |

### 1.3 ring_log.cc 中的硬编码值

| 配置项 | 当前值 | 位置 | 说明 | 建议配置化 |
|--------|--------|------|------|-----------|
| `save_iter % 300` | 300 | 第273行 | futimens调用间隔 | ✅ 是 |
| `sleep(1)` | 1 | 第278行 | 日志写入线程睡眠时间（秒） | ✅ 是 |
| `dash_count < 8` | 8 | 第100行 | Pod名称解析阈值 | ✅ 是 |
| 缓冲区大小 | 512, 256, 80, 64, 32 | 多处 | 各种字符串缓冲区大小 | ⚠️ 可选 |

### 1.4 环境变量依赖

| 环境变量 | 当前默认值 | 位置 | 说明 |
|----------|-----------|------|------|
| `NCCL_MEGATRACE_ENABLE` | 1 | ring_log.cc:21 | 启用/禁用日志 |
| `NCCL_MEGATRACE_LOG_PATH` | "./logs" | ring_log.cc:22 | 日志文件路径 |
| `NCCL_MEGATRACE_SENSTIME` | 3000 | ring_log.cc:23 | 批量刷写灵敏度（毫秒） |
| `POD` | "unknown" | ring_log.cc:177 | Pod名称 |
| `MY_POD_IP` | "unknown" | ring_log.cc:183 | Pod IP地址 |
| `TRAIN_JOB_ID` | "unknown" | ring_log.cc:194 | 训练任务ID |
| `OMPI_COMM_WORLD_RANK` / `RANK` | "0" | log.h:59-61 | Rank编号 |
| `MEGATRACE_LOG_LEVEL` | LOG_ERROR | log.h:25 | 日志级别 |

## 二、Python 分析代码中的硬编码值

### 2.1 trace/analysis/config.py

| 配置项 | 当前值 | 说明 | 建议配置化 |
|--------|--------|------|-----------|
| `timeout_threshold` | 300秒 | Hang检测超时阈值 | ✅ 是 |
| `operation_interval_threshold` | 60秒 | 操作间隔异常检测 | ✅ 是 |
| `collective_hang_threshold` | 120秒 | 集合操作hang阈值 | ✅ 是 |
| `send_recv_hang_threshold` | 180秒 | 发送接收hang阈值 | ✅ 是 |
| `stream_inactive_threshold` | 300秒 | 流无活动阈值 | ✅ 是 |
| `slow_threshold` | 10秒 | 慢操作阈值 | ✅ 是 |
| `operation_performance_thresholds` | 各种值 | 各操作性能阈值 | ✅ 是 |
| `data_size_thresholds` | 1KB-100MB | 数据大小分类阈值 | ✅ 是 |
| `max_file_size_mb` | 1000MB | 最大文件大小 | ✅ 是 |
| `max_lines` | 1000000 | 最大行数 | ✅ 是 |
| `thread_pool_size` | 4 | 线程池大小 | ✅ 是 |

### 2.2 trace/analysis/parallel_slow_detector.py 和 group_hash_slow_detector.py

| 配置项 | 当前值 | 说明 | 建议配置化 |
|--------|--------|------|-----------|
| `GRUBBS_CRITICAL_VALUES` | 硬编码字典 | Grubbs测试临界值 | ⚠️ 可选（统计值） |

### 2.3 bot/alert-to-feishu/config.py

| 配置项 | 当前值 | 说明 | 建议配置化 |
|--------|--------|------|-----------|
| `ALERTMANAGER_TIMEOUT` | 10秒 | AlertManager超时 | ✅ 是（已有环境变量） |
| `ALERTMANAGER_MAX_RETRIES` | 3 | 最大重试次数 | ✅ 是 |
| `ALERTMANAGER_POLL_INTERVAL` | 60秒 | 轮询间隔 | ✅ 是 |
| `RECURRENCE_INTERVAL_SECONDS` | 1200秒 | 复发告警间隔 | ✅ 是 |
| `REPEAT_NOTIFICATION_INTERVAL_SECONDS` | 86400秒 | 重复推送间隔 | ✅ 是 |
| `DATABASE_BACKUP_DAYS` | 7天 | 数据库备份保留天数 | ✅ 是 |
| `LOG_MAX_SIZE` | 100MB | 日志文件最大大小 | ✅ 是 |
| `LOG_BACKUP_COUNT` | 5 | 日志备份数量 | ✅ 是 |
| `MAX_MEMORY_MB` | 1024MB | 最大内存使用量 | ✅ 是 |
| `MAX_CPU_PERCENT` | 90% | 最大CPU使用率 | ✅ 是 |

## 三、配置化方案设计

### 3.1 统一配置架构

```
Megatrace/
├── config/
│   ├── intercept_config.h          # C/C++拦截器配置头文件
│   ├── intercept_config.yaml       # 拦截器配置YAML（编译时生成.h）
│   ├── analysis_config.yaml         # Python分析配置
│   └── alert_config.yaml            # 告警系统配置
├── scripts/
│   └── generate_config.h            # 从YAML生成C头文件的脚本
└── .env.example                     # 环境变量示例文件
```

### 3.2 C/C++ 配置化方案

#### 方案A：编译时配置（推荐）
- 使用CMake或构建脚本从YAML生成C头文件
- 优点：性能最优，无运行时开销
- 缺点：需要重新编译

#### 方案B：运行时配置
- 通过环境变量或配置文件读取
- 优点：无需重新编译
- 缺点：有运行时开销

**推荐采用混合方案**：
- 关键性能参数（缓冲区大小等）使用编译时配置
- 运行时参数（日志路径、日志级别等）使用环境变量或配置文件

### 3.3 Python 配置化方案

- 统一使用YAML配置文件
- 支持环境变量覆盖
- 提供配置验证和默认值机制

## 四、实施计划

### 阶段1：创建配置文件和结构

1. **创建配置目录结构**
   ```
   config/
   ├── intercept_config.yaml
   ├── analysis_config.yaml
   └── alert_config.yaml
   ```

2. **创建配置加载模块**
   - C/C++: `config_loader.h` / `config_loader.c`
   - Python: `config_loader.py`

### 阶段2：C/C++代码重构

1. **替换硬编码宏定义**
   - 将 `ring_log.h` 中的宏改为从配置文件读取
   - 创建配置结构体

2. **环境变量统一管理**
   - 创建统一的环境变量读取函数
   - 提供默认值机制

3. **库搜索列表配置化**
   - 将硬编码的库名称列表移到配置文件

### 阶段3：Python代码重构

1. **统一配置加载**
   - 所有模块使用统一的配置加载器
   - 支持配置文件和环境变量

2. **配置验证**
   - 添加配置验证逻辑
   - 提供清晰的错误信息

### 阶段4：文档和示例

1. **创建配置示例文件**
2. **更新README文档**
3. **添加配置说明**

## 五、配置项详细清单

### 5.1 拦截器配置 (intercept_config.yaml)

```yaml
# 环形缓冲区配置
ring_buffer:
  size: 10000                    # 缓冲区容量
  batch_size: 10240              # 批处理大小
  log_max_len: 256               # 日志条目最大长度

# 日志写入配置
log_writer:
  flush_interval_ms: 4000        # 刷新间隔（毫秒）
  sleep_interval_sec: 1          # 睡眠间隔（秒）
  futimens_interval: 300         # futimens调用间隔
  rotate_size_bytes: 1048576     # 轮转大小（1MB）
  max_versions: 3                # 最大版本数

# 线程初始化配置
thread_init:
  startup_wait_us: 10000         # 启动等待时间（微秒）
  spin_count: 1000               # 自旋等待次数

# 库搜索配置
library_search:
  torch_libs:
    - "libtorch_cuda.so"
    - "libtorch_cuda.so.1"
    - "libtorch_cuda.so.2"
    - "libtorch_python.so"
    - "libtorch_python.so.1"
    - "libtorch_python.so.2"
    - "libnccl.so.2"
    - "libnccl.so.3"
    - "libnccl.so"

# Pod名称解析配置
pod_name:
  min_dash_count: 8              # 最小dash数量

# 环境变量默认值
env_defaults:
  enable: 1                      # NCCL_MEGATRACE_ENABLE
  log_path: "./logs"            # NCCL_MEGATRACE_LOG_PATH
  sensitive_time_ms: 3000       # NCCL_MEGATRACE_SENSTIME
  log_level: "ERROR"            # MEGATRACE_LOG_LEVEL
```

### 5.2 分析配置 (analysis_config.yaml)

```yaml
# Hang分析配置
hang_analysis:
  timeout_threshold: 300         # 超时阈值（秒）
  operation_interval_threshold: 60  # 操作间隔阈值（秒）
  collective_hang_threshold: 120   # 集合操作hang阈值（秒）
  send_recv_hang_threshold: 180    # 发送接收hang阈值（秒）
  stream_inactive_threshold: 300   # 流无活动阈值（秒）

# 慢分析配置
slow_analysis:
  slow_threshold: 10             # 慢操作阈值（秒）
  operation_thresholds:
    AllReduce: 5
    AllGather: 8
    Broadcast: 3
    Send: 15
    Recv: 15
  data_size_thresholds:
    small: 1024                  # 1KB
    medium: 1048576              # 1MB
    large: 10485760              # 10MB
    huge: 104857600              # 100MB

# 分析深度配置
analysis_depth:
  max_file_size_mb: 1000
  max_lines: 1000000
  enable_multithreading: false
  thread_pool_size: 4
  analyze_compressed: true
  enable_realtime_analysis: false

# 输出配置
output:
  format: "text"                 # text, json, csv
  output_file: "distributed_analysis_results.txt"
  save_detailed_logs: true
  log_file: "distributed_log_analysis.log"
  generate_performance_report: true
  performance_report_file: "performance_report.txt"
```

### 5.3 告警配置 (alert_config.yaml)

```yaml
# AlertManager配置
alertmanager:
  url: ""
  timeout: 10
  max_retries: 3
  poll_interval: 60

# 飞书通知配置
feishu:
  group_notification:
    enabled: false
    p0_enabled: true
    p1_enabled: true
    p2_enabled: true
    p3_enabled: false
  table_notification:
    enabled: true
  timeout: 10

# 重复推送配置
repeat_notification:
  enabled: true
  recurrence_interval_seconds: 1200
  repeat_interval_seconds: 86400

# 数据库配置
database:
  path: "./data/alerts.db"
  backup_path: "./data/backup/"
  backup_days: 7

# 日志配置
logging:
  file: "./logs/alert_system.log"
  level: "INFO"
  max_size_mb: 100
  backup_count: 5

# 系统配置
system:
  daemon_mode: false
  pid_file: "./alert_system.pid"
  max_memory_mb: 1024
  max_cpu_percent: 90
```

## 六、环境变量映射表

| 环境变量 | 配置项路径 | 默认值 | 说明 |
|----------|-----------|--------|------|
| `NCCL_MEGATRACE_ENABLE` | intercept.env_defaults.enable | 1 | 启用/禁用日志 |
| `NCCL_MEGATRACE_LOG_PATH` | intercept.env_defaults.log_path | "./logs" | 日志路径 |
| `NCCL_MEGATRACE_SENSTIME` | intercept.env_defaults.sensitive_time_ms | 3000 | 灵敏度时间 |
| `MEGATRACE_LOG_LEVEL` | intercept.env_defaults.log_level | "ERROR" | 日志级别 |
| `POD` | - | "unknown" | Pod名称 |
| `MY_POD_IP` | - | "unknown" | Pod IP |
| `TRAIN_JOB_ID` | - | "unknown" | 训练任务ID |
| `RANK` / `OMPI_COMM_WORLD_RANK` | - | "0" | Rank编号 |
| `ALERTMANAGER_URL` | alert.alertmanager.url | "" | AlertManager地址 |
| `ALERTMANAGER_TIMEOUT` | alert.alertmanager.timeout | 10 | 超时时间 |

## 七、实施优先级

### 高优先级（立即实施）
1. ✅ 环境变量统一管理和默认值
2. ✅ Python分析配置统一化
3. ✅ 关键性能参数配置化（缓冲区大小、批处理大小等）

### 中优先级（近期实施）
1. ⚠️ C/C++编译时配置生成
2. ⚠️ 库搜索列表配置化
3. ⚠️ 配置验证和错误处理

### 低优先级（后续优化）
1. 📝 配置文档完善
2. 📝 配置管理工具
3. 📝 配置热重载（如需要）

## 八、注意事项

1. **向后兼容性**：保持现有环境变量的兼容性
2. **性能影响**：C/C++配置读取应尽可能高效
3. **配置验证**：所有配置项应有合理的默认值和验证
4. **文档同步**：配置变更应及时更新文档
5. **测试覆盖**：新增配置项应有相应的测试

## 九、参考实现

### 9.1 C/C++配置加载示例

```c
// config_loader.h
typedef struct {
    int ring_buffer_size;
    int batch_size;
    int log_max_len;
    int flush_interval_ms;
    // ... 其他配置项
} intercept_config_t;

intercept_config_t* load_intercept_config(const char* config_path);
```

### 9.2 Python配置加载示例

```python
# config_loader.py
import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config = self._load_yaml(config_path)
        self._apply_env_overrides()
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        # 应用环境变量覆盖
        pass
    
    def get(self, key_path: str, default=None):
        # 支持点号分隔的路径访问
        pass
```

---

**文档版本**: 1.0  
**最后更新**: 2024-01-XX  
**维护者**: Megatrace Team

