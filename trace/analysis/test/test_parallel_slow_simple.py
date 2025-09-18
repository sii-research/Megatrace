#!/usr/bin/env python3
"""
Simple test script for parallel slow detection functionality (kept for benchmarking/dev use).
Prefer: python analysis/test_analyzer.py --log-path analysis/logs --test-type slow --config-path analysis/config.yaml
"""

import os


def main():
    print("Simple Parallel Slow Detection Test (use unified CLI in production)")
    print("=" * 40)

    logs_path = "analysis/logs/"
    config_file = "analysis/config.yaml"
    if not os.path.exists(logs_path):
        print(f"Error: Logs directory '{logs_path}' not found!")
        return
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found!")
        return

    try:
        import sys
        sys.path.insert(0, 'analysis')
        from parallel_slow_detector import ParallelSlowDetector

        detector = ParallelSlowDetector(logs_path, verbose=True, use_multiprocessing=False)
        results = detector.analyze_parallel_slow_nodes()
        if results:
            detector.print_results(results)
        else:
            print("Analysis failed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
