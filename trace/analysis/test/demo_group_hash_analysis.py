#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于GroupHash的慢速检测器演示脚本
展示如何在实际环境中使用该工具
"""

import os
import sys
import argparse

# 添加analysis目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from group_hash_slow_detector import GroupHashSlowDetector

def demo_basic_usage():
    """演示基本用法"""
    print("=== 基本用法演示 ===")
    
    # 检查是否有日志文件
    logs_path = "analysis/logs/log_new"
    if not os.path.exists(logs_path):
        print(f"日志路径不存在: {logs_path}")
        print("请确保日志文件存在于该路径下")
        return
    
    # 创建检测器
    detector = GroupHashSlowDetector(
        logs_path=logs_path,
        verbose=True,
        use_multiprocessing=True
    )
    
    # 运行分析
    print(f"\n分析日志路径: {logs_path}")
    results = detector.run_analysis()
    
    if results:
        print(f"\n✅ 分析完成！共发现 {len(results)} 个通信组")
        
        # 显示一些统计信息
        total_operations = sum(len(ops) for ops in detector.operations_by_group.values())
        total_ranks = len(set().union(*[gp.ranks for gp in detector.group_performance.values()]))
        
        print(f"总操作数: {total_operations}")
        print(f"涉及rank数: {total_ranks}")
        
        # 找出最慢的rank
        rank_slow_totals = {}
        for rank, group_counts in detector.rank_slow_counts.items():
            rank_slow_totals[rank] = sum(group_counts.values())
        
        if rank_slow_totals:
            max_slow_rank = max(rank_slow_totals.items(), key=lambda x: x[1])
            print(f"最慢的rank: {max_slow_rank[0]} (慢速操作 {max_slow_rank[1]} 次)")
    else:
        print("❌ 分析失败或未找到有效数据")

def demo_advanced_analysis():
    """演示高级分析功能"""
    print("\n=== 高级分析功能演示 ===")
    
    logs_path = "analysis/logs/log_new"
    if not os.path.exists(logs_path):
        print(f"日志路径不存在: {logs_path}")
        return
    
    detector = GroupHashSlowDetector(
        logs_path=logs_path,
        verbose=False,  # 关闭详细输出
        use_multiprocessing=True
    )
    
    # 分步执行分析
    print("1. 解析日志文件...")
    operations_by_group = detector.parse_all_logs()
    
    if not operations_by_group:
        print("未找到操作数据")
        return
    
    print(f"   发现 {len(operations_by_group)} 个通信组")
    
    print("2. 分析性能...")
    group_performance = detector.analyze_group_performance()
    
    if not group_performance:
        print("性能分析失败")
        return
    
    print(f"   完成 {len(group_performance)} 个组的性能分析")
    
    # 自定义分析
    print("3. 自定义分析...")
    
    # 按函数类型分组
    function_groups = {}
    for group_hash, group_perf in group_performance.items():
        for func in set(group_perf.functions):
            if func not in function_groups:
                function_groups[func] = []
            function_groups[func].append(group_perf)
    
    print(f"   按函数类型分组:")
    for func, groups in function_groups.items():
        print(f"     {func}: {len(groups)} 个组")
    
    # 按数据大小分组
    size_groups = {}
    for group_hash, group_perf in group_performance.items():
        for size in set(group_perf.data_sizes):
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(group_perf)
    
    print(f"   按数据大小分组:")
    for size, groups in sorted(size_groups.items()):
        print(f"     {size} bytes: {len(groups)} 个组")
    
    # 生成矩阵数据
    print("4. 生成矩阵数据...")
    matrix_data = detector.generate_slow_rank_matrix()
    
    if matrix_data:
        print(f"   矩阵大小: {len(matrix_data)} x {len(matrix_data[0])}")
        
        # 导出为CSV格式
        csv_file = "slow_rank_matrix.csv"
        try:
            import csv
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                all_groups = sorted(detector.group_mapping.values())
                header = ['Rank'] + [f'Group_{group_id}' for group_id in all_groups]
                writer.writerow(header)
                # 写入数据
                writer.writerows(matrix_data)
            print(f"   矩阵已导出到: {csv_file}")
        except ImportError:
            print("   无法导出CSV (缺少csv模块)")
    
    print("✅ 高级分析完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于GroupHash的慢速检测器演示')
    parser.add_argument('--basic', action='store_true', help='运行基本用法演示')
    parser.add_argument('--advanced', action='store_true', help='运行高级分析演示')
    parser.add_argument('--all', action='store_true', help='运行所有演示')
    
    args = parser.parse_args()
    
    if not any([args.basic, args.advanced, args.all]):
        # 默认运行基本演示
        args.basic = True
    
    try:
        if args.basic or args.all:
            demo_basic_usage()
        
        if args.advanced or args.all:
            demo_advanced_analysis()
            
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
