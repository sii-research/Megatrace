# Megatrace 配置使用指南

本文档说明如何使用Megatrace的配置文件系统。

## 配置文件结构

```
Megatrace/
├── config/
│   ├── intercept_config.yaml.example    # 拦截器配置示例
│   ├── analysis_config.yaml.example     # 分析模块配置示例
│   └── alert_config.yaml.example        # 告警系统配置示例
└── CONFIGURATION_PLAN.md                # 配置化方案详细文档
```

## 快速开始

### 1. 拦截器配置

拦截器配置主要用于C/C++代码，目前主要通过环境变量控制。未来版本将支持YAML配置文件。

**当前使用方式（环境变量）：**

```bash
# 启用日志记录
export NCCL_MEGATRACE_ENABLE=1

# 设置日志路径
export NCCL_MEGATRACE_LOG_PATH=/path/to/logs

# 设置批量刷写灵敏度（毫秒）
export NCCL_MEGATRACE_SENSTIME=3000

# 设置日志级别
export MEGATRACE_LOG_LEVEL=INFO
```

**环境变量说明：**

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `NCCL_MEGATRACE_ENABLE` | 1 | 启用/禁用日志（1=启用，0=禁用） |
| `NCCL_MEGATRACE_LOG_PATH` | "./logs" | 日志文件保存路径 |
| `NCCL_MEGATRACE_SENSTIME` | 3000 | 批量刷写灵敏度（毫秒） |
| `MEGATRACE_LOG_LEVEL` | ERROR | 日志级别（ERROR/WARN/INFO/DEBUG） |
| `POD` | "unknown" | Pod名称（Kubernetes环境） |
| `MY_POD_IP` | "unknown" | Pod IP地址（Kubernetes环境） |
| `TRAIN_JOB_ID` | "unknown" | 训练任务ID |
| `RANK` / `OMPI_COMM_WORLD_RANK` | "0" | Rank编号 |

### 2. 分析模块配置

分析模块使用YAML配置文件。

**使用步骤：**

1. 复制示例配置文件：
```bash
cp config/analysis_config.yaml.example config/analysis_config.yaml
```

2. 根据需要修改配置：
```yaml
# 修改Hang检测阈值
hang_analysis:
  timeout_threshold: 300  # 5分钟

# 修改慢操作阈值
slow_analysis:
  slow_threshold: 10  # 10秒
```

3. 在代码中使用配置：
```python
from config_loader import ConfigLoader

config = ConfigLoader('config/analysis_config.yaml')
timeout = config.get('hang_analysis.timeout_threshold', default=300)
```

**主要配置项：**

- **Hang分析配置**：超时阈值、操作间隔阈值等
- **慢分析配置**：慢操作阈值、操作性能阈值、数据大小阈值等
- **输出配置**：输出格式、文件路径等
- **分析深度配置**：文件大小限制、线程池大小等

### 3. 告警系统配置

告警系统使用YAML配置文件，同时支持环境变量覆盖。

**使用步骤：**

1. 复制示例配置文件：
```bash
cp config/alert_config.yaml.example config/alert_config.yaml
```

2. 配置AlertManager和飞书webhook：
```yaml
alertmanager:
  url: "http://your-alertmanager:9093"

feishu:
  group_notification:
    webhooks:
      p0: "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
```

3. 环境变量覆盖（可选）：
```bash
export ALERTMANAGER_URL=http://localhost:9093
export LOG_LEVEL=INFO
```

**配置优先级：**

环境变量 > 配置文件 > 默认值

## 配置项详细说明

### 拦截器配置项

#### 环形缓冲区配置
- `ring_buffer.size`: 缓冲区容量（默认：10000）
- `ring_buffer.batch_size`: 批处理大小（默认：10240）
- `ring_buffer.log_max_len`: 日志条目最大长度（默认：256字节）

#### 日志写入配置
- `log_writer.flush_interval_ms`: 刷新间隔（默认：4000毫秒）
- `log_writer.sleep_interval_sec`: 睡眠间隔（默认：1秒）
- `log_writer.rotate_size_bytes`: 日志轮转大小（默认：1MB）
- `log_writer.max_versions`: 最大版本数（默认：3）

### 分析模块配置项

#### Hang分析
- `hang_analysis.timeout_threshold`: 超时阈值（默认：300秒）
- `hang_analysis.operation_interval_threshold`: 操作间隔阈值（默认：60秒）
- `hang_analysis.collective_hang_threshold`: 集合操作hang阈值（默认：120秒）

#### 慢分析
- `slow_analysis.slow_threshold`: 慢操作阈值（默认：10秒）
- `slow_analysis.operation_performance_thresholds`: 各操作性能阈值
- `slow_analysis.data_size_thresholds`: 数据大小分类阈值

### 告警系统配置项

#### AlertManager
- `alertmanager.url`: AlertManager API地址
- `alertmanager.timeout`: 超时时间（默认：10秒）
- `alertmanager.max_retries`: 最大重试次数（默认：3）

#### 飞书通知
- `feishu.group_notification.enabled`: 群组通知总开关
- `feishu.group_notification.webhooks`: 各等级webhook配置
- `feishu.table_notification.enabled`: 表格通知开关

## 配置验证

### Python配置验证

告警系统配置包含验证功能：

```python
from bot.alert_to_feishu.config import validate_config

if not validate_config():
    print("配置验证失败，请检查配置文件")
    exit(1)
```

### 环境变量检查

可以使用以下脚本检查环境变量：

```bash
#!/bin/bash
echo "检查Megatrace环境变量配置..."
echo "NCCL_MEGATRACE_ENABLE: ${NCCL_MEGATRACE_ENABLE:-未设置（默认：1）}"
echo "NCCL_MEGATRACE_LOG_PATH: ${NCCL_MEGATRACE_LOG_PATH:-未设置（默认：./logs）}"
echo "MEGATRACE_LOG_LEVEL: ${MEGATRACE_LOG_LEVEL:-未设置（默认：ERROR）}"
```

## 最佳实践

1. **开发环境**：使用默认配置或最小配置
2. **生产环境**：根据实际需求调整阈值和路径
3. **Kubernetes环境**：通过ConfigMap管理配置文件
4. **配置版本控制**：将配置文件纳入版本控制，但敏感信息（如webhook）使用环境变量

## 故障排查

### 配置未生效

1. 检查配置文件路径是否正确
2. 检查环境变量是否设置
3. 检查配置文件格式是否正确（YAML语法）
4. 查看日志文件中的配置加载信息

### 配置值异常

1. 检查配置值的类型是否正确（字符串/数字/布尔值）
2. 检查配置值的范围是否合理
3. 查看配置验证错误信息

## 迁移指南

### 从硬编码迁移到配置文件

1. 识别代码中的硬编码值
2. 在配置文件中添加对应配置项
3. 修改代码读取配置而非使用硬编码
4. 提供默认值以保持向后兼容
5. 更新文档和示例

## 相关文档

- [配置化方案详细文档](CONFIGURATION_PLAN.md)
- [拦截器README](trace/intercept/README.md)
- [分析模块README](trace/analysis/README.md)
- [告警系统README](bot/README.md)

## 贡献

如果您发现需要配置化的硬编码值，请：

1. 在 `CONFIGURATION_PLAN.md` 中记录
2. 创建Issue说明配置需求
3. 提交PR实现配置化

---

**最后更新**: 2024-01-XX

