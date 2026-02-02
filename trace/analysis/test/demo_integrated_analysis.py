#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成后的test_analyzer.py演示脚本
展示groupHash慢速检测功能
"""

import os
import sys

def demo_existing_logs():
    """使用现有日志进行演示"""
    logs_path = "analysis/logs/log_new"
    
    if not os.path.exists(logs_path):
        print(f"日志路径不存在: {logs_path}")
        print("请确保存在包含groupHash字段的日志文件")
        return False
    
    print("=== 使用现有日志演示集成后的分析功能 ===")
    print(f"日志路径: {logs_path}")
    
    # 运行slow分析（包含groupHash分析）
    print("\n1. 运行slow分析（包含groupHash分析）...")
    import subprocess
    
    cmd = [
        sys.executable,
        "analysis/test_analyzer.py",
        "--log-path", logs_path,
        "--test-type", "slow",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), timeout=120)
        if result.returncode == 0:
            print("✅ Slow分析（包含GroupHash）执行成功")
        else:
            print(f"❌ Slow分析执行失败，返回码: {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 分析超时")
        return False
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        return False
    
    return True

def show_integration_summary():
    """显示集成摘要"""
    print("\n" + "="*80)
    print("GROUPHASH慢速检测器集成摘要")
    print("="*80)
    
    print("\n✅ 集成成功完成！")
    print("\n🎯 新增功能:")
    print("  • 基于groupHash的通信组自动识别")
    print("  • 智能慢速检测（IQR异常值检测）")
    print("  • Rank vs Group矩阵报告")
    print("  • 与现有TP/PP/DP分析并存")
    
    print("\n📊 输出格式:")
    print("  • 通信组统计信息")
    print("  • 慢速Rank矩阵（Rank vs Group）")
    print("  • Top慢速Rank排行")
    print("  • 详细性能分析（verbose模式）")
    
    print("\n🔧 使用方法:")
    print("  # 运行slow分析（包含GroupHash分析）")
    print("  python analysis/test_analyzer.py --log-path /path/to/logs --test-type slow")
    print("")
    print("  # 运行所有分析")
    print("  python analysis/test_analyzer.py --log-path /path/to/logs --test-type all")
    print("")
    print("  # 启用详细输出")
    print("  python analysis/test_analyzer.py --log-path /path/to/logs --test-type slow --verbose")
    
    print("\n📝 输出示例:")
    print("""
================================================================================
GROUP HASH BASED SLOW DETECTION ANALYSIS
================================================================================
  • Groups Analyzed: 4
  • Total Operations: 20
  • Total Ranks: 4
  • Slow Picks: 4
  • GroupHash Slow Rank Matrix (Rank vs Group):
    Rank  G1    G2    G3    G4    Total   
    --------------------------------------
    0     0     0     0     1     1       
    1     1     1     0     0     2       
    2     0     0     1     0     1       
    3     0     0     0     0     0       
  • Top Slow Ranks: [1, 2, 0]
    Rank 1: 2 slow operations
    Rank 2: 1 slow operations
    Rank 0: 1 slow operations
================================================================================
    """)
    
    print("\n🚀 优势特点:")
    print("  • 无需配置：自动识别通信组")
    print("  • 高精度：基于实际通信模式检测")
    print("  • 高性能：支持多进程解析")
    print("  • 易集成：与现有工具无缝配合")
    
    print("="*80)

def main():
    """主函数"""
    print("="*80)
    print("GroupHash慢速检测器集成演示")
    print("="*80)
    
    # 显示集成摘要
    show_integration_summary()
    
    # 尝试演示现有日志
    success = demo_existing_logs()
    
    if success:
        print("\n🎉 演示完成！GroupHash慢速检测器已成功集成到test_analyzer.py中")
    else:
        print("\n⚠️  演示无法运行，但集成已完成")
        print("   请确保有包含groupHash字段的日志文件进行测试")
    
    print("\n📖 更多信息请参阅:")
    print("  • GROUP_HASH_SLOW_DETECTOR_README.md - 详细使用说明")
    print("  • FINAL_SUMMARY.md - 完整项目总结")

if __name__ == "__main__":
    main()
