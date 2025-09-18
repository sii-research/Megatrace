#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log分析器配置文件
定义各种分析参数和规则
"""

# Log文件扩展名
LOG_EXTENSIONS = {
    '.log', '.txt', '.out', '.err', '.log.gz', '.log.bz2'
}

# 分布式训练log格式模式
DISTRIBUTED_LOG_PATTERNS = {
    # 完整格式: [save_count 5] [1755877674.248008482] [Rank 0] Fun AllReduce Data 25165824 stream 0x562a9d0b5020 opCount 8 groupHash -5956285979098587305
    'full_pattern': r'\[save_count (\d+)\] \[([\d.]+)\] \[Rank (\d+)\] Fun (\w+) Data (\d+) stream (0x[0-9a-fA-F]+) opCount (\d+) groupHash (-?\d+)',
    
    # 简化格式: [1755185996.690626046] [Rank 0] Fun AllReduce Data 1
    'simple_pattern': r'\[([\d.]+)\] \[Rank (\d+)\] Fun (\w+) Data (\d+)',
    
    # 时间戳格式: 1755185996.690626046
    'timestamp_pattern': r'(\d{10,13}\.\d+)',
    
    # Rank格式: [Rank 0]
    'rank_pattern': r'\[Rank (\d+)\]',
    
    # 函数类型: Fun AllReduce
    'function_pattern': r'Fun (\w+)',
    
    # 数据大小: Data 25165824
    'data_size_pattern': r'Data (\d+)',
    
    # 操作计数: opCount 0
    'op_count_pattern': r'opCount (\d+)',
    
    # 组哈希: groupHash -5956285979098587305
    'group_hash_pattern': r'groupHash (-?\d+)'
}

# 分布式训练操作类型
DISTRIBUTED_OPERATIONS = {
    'collective': ['AllReduce', 'AllGather', 'Broadcast', 'Reduce', 'Scatter', 'Gather'],
    'point_to_point': ['Send', 'Recv', 'SendRecv'],
    'synchronization': ['Barrier', 'Wait', 'Synchronize']
}

# Hang分析配置 - 针对分布式训练
HANG_ANALYSIS_CONFIG = {
    # 超时阈值（秒）
    'timeout_threshold': 300,  # 5分钟
    
    # 分布式训练hang检测模式
    'hang_patterns': [
        'timeout', 'deadlock', 'stuck', 'frozen', 'not responding',
        'waiting', 'blocked', 'hung', 'stalled'
    ],
    
    # 操作间隔异常检测
    'operation_interval_threshold': 60,  # 60秒内没有新操作视为可能hang
    
    # 特定操作hang检测
    'collective_hang_threshold': 120,  # 集合操作超过2分钟视为hang
    'send_recv_hang_threshold': 180,   # 发送接收超过3分钟视为hang
    
    # 流状态检测
    'stream_inactive_threshold': 300,   # 流5分钟无活动视为hang
}

# 慢分析配置 - 针对分布式训练
SLOW_ANALYSIS_CONFIG = {
    # 慢操作阈值（秒）
    'slow_threshold': 10,  # 10秒
    
    # 分布式训练慢操作检测
    'slow_operation_patterns': [
        'slow', 'timeout', 'long', 'delay', 'latency',
        'performance', 'bottleneck', 'slow operation'
    ],
    
    # 操作性能分析
    'operation_performance_thresholds': {
        'AllReduce': 5,      # AllReduce超过5秒视为慢
        'AllGather': 8,      # AllGather超过8秒视为慢
        'Broadcast': 3,      # Broadcast超过3秒视为慢
        'Send': 15,          # Send超过15秒视为慢
        'Recv': 15,          # Recv超过15秒视为慢
    },
    
    # 数据大小性能分析
    'data_size_thresholds': {
        'small': 1024,           # 1KB
        'medium': 1024*1024,     # 1MB
        'large': 10*1024*1024,   # 10MB
        'huge': 100*1024*1024    # 100MB
    },
    
    # 性能指标关键词
    'performance_metrics': [
        'response time', 'execution time', 'processing time',
        'duration', 'elapsed time', 'wall time', 'throughput'
    ]
}

# 输出配置
OUTPUT_CONFIG = {
    # 输出文件格式
    'output_format': 'text',  # text, json, csv
    
    # 输出文件路径
    'output_file': 'distributed_analysis_results.txt',
    
    # 是否保存详细日志
    'save_detailed_logs': True,
    
    # 日志文件路径
    'log_file': 'distributed_log_analysis.log',
    
    # 是否生成性能报告
    'generate_performance_report': True,
    
    # 性能报告文件路径
    'performance_report_file': 'performance_report.txt'
}

# 分析深度配置
ANALYSIS_DEPTH_CONFIG = {
    # 最大文件大小（MB）
    'max_file_size_mb': 1000,
    
    # 最大行数
    'max_lines': 1000000,
    
    # 是否启用多线程分析
    'enable_multithreading': False,
    
    # 线程池大小
    'thread_pool_size': 4,
    
    # 是否分析压缩文件
    'analyze_compressed': True,
    
    # 是否启用实时分析
    'enable_realtime_analysis': False
}

# 分布式训练特定配置
DISTRIBUTED_TRAINING_CONFIG = {
    # 是否分析多rank文件
    'analyze_multiple_ranks': True,
    
    # 是否进行rank间对比分析
    'cross_rank_analysis': True,
    
    # 是否检测rank同步问题
    'detect_rank_sync_issues': True,
    
    # 是否分析数据流模式
    'analyze_data_flow_patterns': True,
    
    # 是否检测负载均衡问题
    'detect_load_balancing_issues': True
} 