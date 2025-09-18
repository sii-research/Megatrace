# LSTM Time Series Analysis for Distributed Training Logs

## 概述

这个模块使用LSTM（长短期记忆网络）对分布式训练日志进行时序分析和异常检测。**重要特性：每个rank的log按照stream分组进行LSTM分析**，确保时序分析的准确性和相关性。

主要功能包括：

1. **连续序列检测**：自动识别opCount连续的日志条目序列
2. **Stream分组分析**：**每个rank的log按照stream分组进行LSTM分析**
3. **LSTM时序预测**：使用LSTM模型预测操作执行时间
4. **异常检测**：基于预测误差检测异常执行时间节点
5. **偏离度计算**：计算每个异常节点的偏离程度
6. **多Rank多Stream支持**：为每个rank的每个stream提供独立的异常分析

## 核心特性

### Stream分组分析 ⭐ **重要特性**
- **每个rank的log按照stream分组进行LSTM分析**
- 确保同一stream内的操作具有时序相关性
- 不同stream的操作可能具有不同的执行模式
- 支持多stream的并行分析，每个stream独立建模

### 连续序列识别
- 自动检测save_count之间opCount连续的数据
- 如果不连续，则分开进行LSTM分析
- **按rank和stream分别识别连续序列**

### LSTM模型
- 可配置的序列长度和隐藏层大小
- 自动数据标准化和反标准化
- 支持GPU加速（如果可用）
- 降级到简单移动平均预测（当PyTorch不可用时）

### 异常检测
- 可配置的异常阈值
- 基于相对误差的偏离度计算
- 详细的异常位置和严重程度信息
- **按stream分别报告异常情况**

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `torch>=1.9.0` - PyTorch深度学习框架
- `scikit-learn>=1.0.0` - 数据预处理
- `numpy>=1.20` - 数值计算
- `pandas>=1.3` - 数据处理

## 使用方法

### 1. 基本使用

```python
from lstm_analysis import LSTMTimeSeriesAnalyzer
from analysis import DistributedLogAnalyzer

# 初始化分析器
lstm_analyzer = LSTMTimeSeriesAnalyzer(
    sequence_length=10,  # LSTM序列长度
    threshold=2.0        # 异常检测阈值
)

# 解析日志
log_analyzer = DistributedLogAnalyzer("path/to/logs")
log_analyzer.discover_log_files()
rank_entries = log_analyzer.parse_log_files()

# 执行LSTM分析（自动按rank和stream分组）
results = lstm_analyzer.analyze_logs(rank_entries)

# 生成报告
report = lstm_analyzer.generate_report()
print(report)
```

### 2. 命令行使用

```bash
# 基本分析
python lstm_analysis.py --log_path /path/to/logs

# 自定义参数
python lstm_analysis.py \
    --log_path /path/to/logs \
    --sequence_length 15 \
    --threshold 1.5 \
    --verbose
```

### 3. 获取特定Rank和Stream的异常信息

```python
# 获取Rank 0的所有异常，按stream分组
rank0_anomalies = lstm_analyzer.get_rank_anomalies(0)

for stream, anomalies in rank0_anomalies.items():
    print(f"Stream {stream}: {len(anomalies)} sequences with anomalies")
    for anomaly in anomalies:
        print(f"  Sequence: {anomaly['sequence_info']}")
        print(f"  Anomalies: {anomaly['anomalies']}")

# 获取特定rank和stream的异常
rank0_stream1_anomalies = lstm_analyzer.get_rank_stream_anomalies(0, "0x1234")
```

## 配置参数

### LSTMTimeSeriesAnalyzer
- `sequence_length`: LSTM输入序列长度（默认：10）
- `threshold`: 异常检测阈值，相对误差超过此值视为异常（默认：2.0）

### LSTMAnalyzer
- `sequence_length`: LSTM序列长度（默认：10）
- `hidden_size`: LSTM隐藏层大小（默认：64）
- `num_layers`: LSTM层数（默认：2）
- `threshold`: 异常检测阈值（默认：2.0）

## 输出格式

### 异常检测结果（按Stream分组）
```python
{
    '0x1234': [  # Stream ID
        {
            'sequence_info': {
                'start_save_count': 1,
                'end_save_count': 5,
                'stream': '0x1234',
                'operations_count': 20
            },
            'anomalies': {
                'anomalies_found': 2,
                'max_deviation': 3.5,
                'anomaly_indices': [15, 18],
                'deviation_scores': [2.1, 3.5]
            }
        }
    ],
    '0x5678': [  # Another Stream
        # ... similar structure
    ]
}
```

### 分析报告
报告包含：
- **每个Rank的Stream数量**
- **每个Stream的序列数量**
- **每个序列的异常数量**
- **异常的最大偏离度**
- **异常位置索引**
- **按Stream分组的详细统计**

## 算法原理

### 1. Stream分组和连续序列检测
- **按rank和stream分组日志条目**
- 在每个stream内按save_count和op_count排序
- 检测每个stream内opCount连续的条目序列
- **确保LSTM分析在同一stream内进行，保持时序相关性**

### 2. LSTM时序预测
- 提取每个stream内操作间的时间间隔
- 数据标准化处理
- 为每个stream训练独立的LSTM模型
- 反标准化得到实际时间预测值

### 3. 异常检测
- 计算预测值与实际值的相对误差
- 超过阈值的点标记为异常
- 计算偏离度分数
- **按stream分别报告异常情况**

## 为什么按Stream分组很重要？

### 1. **时序相关性**
- 同一stream内的操作通常具有相似的执行模式
- 不同stream可能代表不同的计算任务或数据流
- 混合分析可能导致模型学习到无关的模式

### 2. **执行模式差异**
- 不同stream可能有不同的性能特征
- 某些stream可能更敏感于系统负载变化
- 按stream分组可以识别stream特定的异常模式

### 3. **调试和优化**
- 可以定位到具体哪个stream出现了问题
- 有助于针对特定stream进行性能优化
- 提供更精确的异常定位信息

## 性能优化

### GPU加速
- 自动检测CUDA可用性
- 模型和数据自动转移到GPU
- 显著提升训练和推理速度

### 内存优化
- 批处理训练
- 早期停止机制
- 可配置的序列长度
- **按stream分别处理，避免内存溢出**

## 故障排除

### 常见问题

1. **PyTorch不可用**
   - 系统会自动降级到简单移动平均预测
   - 功能仍然可用，但精度可能降低

2. **内存不足**
   - 减少sequence_length参数
   - 减少hidden_size参数
   - **按stream分别处理，减少内存占用**

3. **训练收敛慢**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量
   - **确保同一stream内的数据具有时序相关性**

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细输出
analyzer = LSTMTimeSeriesAnalyzer(verbose=True)
```

## 扩展功能

### 自定义异常检测
```python
class CustomAnomalyDetector:
    def detect_anomalies(self, actual, predicted):
        # 实现自定义异常检测逻辑
        pass

analyzer.lstm_analyzer._detect_anomalies = CustomAnomalyDetector().detect_anomalies
```

### 自定义LSTM模型
```python
class CustomLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义模型架构
        
analyzer.lstm_analyzer.model = CustomLSTMModel()
```

### Stream特定的分析参数
```python
# 可以为不同stream设置不同的参数
stream_params = {
    '0x1234': {'sequence_length': 8, 'threshold': 1.5},
    '0x5678': {'sequence_length': 12, 'threshold': 2.5}
}
```

## 测试

运行测试套件：
```bash
python test_lstm_analysis.py
```

测试覆盖：
- 连续序列检测
- **Stream分组功能**
- 异常结果处理
- LSTM分析器
- 完整分析流程

## 贡献

欢迎提交Issue和Pull Request来改进这个模块。主要改进方向：
- 新的异常检测算法
- **Stream分组策略优化**
- 性能优化
- 更多可视化功能
- 支持更多日志格式
