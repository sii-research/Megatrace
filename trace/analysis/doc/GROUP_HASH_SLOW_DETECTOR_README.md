# 基于GroupHash的慢速检测器

## 概述
这是一个专门为分析分布式训练日志中基于`groupHash`字段的通信组性能差异而设计的工具。它能够：

1. 根据`groupHash`值自动识别通信组
2. 分析每个通信组内各rank的性能差异
3. 检测慢速rank并统计慢速次数
4. 生成清晰的矩阵报告，横向显示每个rank，纵向显示每个组

## 新特性
- **基于groupHash的通信组识别**：不再依赖固定的TP/PP/DP配置，而是根据日志中的`groupHash`字段自动识别通信组
- **智能慢速检测**：使用IQR（四分位距）方法检测异常值，识别慢速rank
- **矩阵化报告**：生成清晰的矩阵，显示每个rank在每个组中的慢速次数
- **向后兼容**：支持新的日志格式，同时保持对旧格式的兼容性

## 日志格式要求
支持以下格式的日志：
```
[save_count 5] [1755877674.248008482] [Rank 0] Fun AllReduce Data 25165824 stream 0x562a9d0b5020 opCount 8 groupHash -5956285979098587305
```

**必需字段**：
- `save_count`：保存计数
- `timestamp`：时间戳
- `Rank`：进程排名
- `Fun`：函数名称
- `Data`：数据大小
- `stream`：流指针
- `opCount`：操作计数
- `groupHash`：组哈希值（用于识别通信组）

## 安装依赖
```bash
pip install numpy pyyaml
```

## 使用方法

### 1. 命令行使用
```bash
# 基本用法
python analysis/group_hash_slow_detector.py /path/to/logs

# 启用详细输出
python analysis/group_hash_slow_detector.py /path/to/logs --verbose

# 禁用多进程（用于调试）
python analysis/group_hash_slow_detector.py /path/to/logs --no-mp
```

### 2. 编程接口使用
```python
from analysis.group_hash_slow_detector import GroupHashSlowDetector

# 创建检测器
detector = GroupHashSlowDetector(
    logs_path="/path/to/logs",
    verbose=True,
    use_multiprocessing=True
)

# 运行完整分析
results = detector.run_analysis()

# 或者分步执行
detector.parse_all_logs()
detector.analyze_group_performance()
detector.print_slow_rank_summary()
```

## 输出示例

### 通信组信息
```
Found 4 communication groups:
  Group 1: 4 ranks, 4 operations
    Ranks: [0, 1, 2, 3]
    Functions: ['Broadcast']
    Data sizes: [32768]
    Slowest rank: 1 (time: 1755877672.517948)
    Outliers detected: 1 ranks above threshold
```

### 慢速Rank矩阵
```
SLOW RANK MATRIX (Rank vs Group)
--------------------------------------------------------------------------------
Rank   Group_1 Group_2 Group_3 Group_4
--------------------------------------------------------------------------------
0      0       0       0       1      
1      1       1       0       0      
2      0       0       1       0      
3      0       0       0       0      
```

### 矩阵说明
- **行**：每个rank（进程）
- **列**：每个通信组（按顺序编号）
- **数值**：该rank在该组中慢速操作的次数
- **0**：表示该rank在该组中表现正常
- **>0**：表示该rank在该组中慢速操作的次数

## 核心算法

### 1. 通信组识别
- 解析所有日志文件，提取包含`groupHash`的操作
- 按`groupHash`值分组，相同hash值的操作属于同一通信组
- 为每个组分配顺序ID（Group_1, Group_2, ...）

### 2. 慢速检测
- 对每个通信组，收集所有rank的操作时间戳
- 使用IQR方法计算异常值阈值：`Q3 + 1.5 * IQR`
- 超过阈值的rank被标记为慢速
- 统计每个rank在每个组中的慢速次数

### 3. 性能分析
- 计算每个组的统计信息（Q1, Q3, IQR, 异常值数量）
- 识别最慢的rank和对应的延迟时间
- 分析操作类型和数据大小分布

## 配置选项

### 初始化参数
- `logs_path`：日志文件目录路径
- `verbose`：是否启用详细输出（默认False）
- `use_multiprocessing`：是否使用多进程（默认True）

### 性能优化
- **多进程解析**：自动检测CPU核心数，并行解析多个日志文件
- **内存管理**：按需加载和释放数据，支持大日志文件
- **缓存机制**：缓存解析结果，避免重复计算

## 故障排除

### 常见问题
1. **"No log files found"**
   - 检查日志路径是否正确
   - 确认日志文件扩展名为`.log`

2. **"Could not extract rank from filename"**
   - 检查日志文件名格式
   - 支持的模式：`rank_0.log`, `0.log`, `log_0.txt`, `rank0.log`

3. **"No operations with groupHash found"**
   - 确认日志包含`groupHash`字段
   - 检查正则表达式是否匹配日志格式

### 调试技巧
- 使用`--verbose`参数查看详细输出
- 使用`--no-mp`参数禁用多进程，便于调试
- 检查日志文件编码（推荐UTF-8）

## 扩展功能

### 自定义分析
```python
# 获取特定组的性能数据
group_perf = detector.group_performance[group_hash]

# 分析特定rank的性能
rank_slow_counts = detector.rank_slow_counts[rank]

# 生成自定义报告
matrix_data = detector.generate_slow_rank_matrix()
```

### 数据导出
```python
# 导出为CSV格式
import csv
matrix_data = detector.generate_slow_rank_matrix()
with open('slow_rank_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Rank'] + [f'Group_{i}' for i in range(1, len(matrix_data[0]))])
    writer.writerows(matrix_data)
```

## 版本历史
- **v1.0**：初始版本，支持基于groupHash的通信组分析
- 支持IQR异常值检测
- 生成矩阵化慢速rank报告
- 多进程日志解析支持

## 贡献指南
欢迎提交问题报告和功能建议。请确保：
1. 测试代码在您的环境中正常工作
2. 提供详细的错误信息和复现步骤
3. 遵循现有的代码风格和架构

## 许可证
本项目遵循与主项目相同的许可证条款。
