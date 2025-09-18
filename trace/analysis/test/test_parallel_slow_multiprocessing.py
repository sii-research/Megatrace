#!/usr/bin/env python3
"""
Benchmark script for multiprocessing vs single-threaded parsing in ParallelSlowDetector.
Prefer: unified CLI for functional runs; keep this for perf comparison.
"""

import os
import time


def main():
    print("Parallel Slow Detection - Multiprocessing Benchmark")
    print("=" * 50)

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

        print("Testing single-threaded version...")
        start_time = time.time()
        detector_single = ParallelSlowDetector(logs_path, verbose=False, use_multiprocessing=False)
        results_single = detector_single.analyze_parallel_slow_nodes()
        single_time = time.time() - start_time
        print(f"Single-threaded completed in {single_time:.2f} seconds")

        print("\nTesting multiprocessing version...")
        start_time = time.time()
        detector_multi = ParallelSlowDetector(logs_path, verbose=False, use_multiprocessing=True)
        results_multi = detector_multi.analyze_parallel_slow_nodes()
        multi_time = time.time() - start_time
        print(f"Multiprocessing completed in {multi_time:.2f} seconds")

        if results_single and results_multi:
            print("\nPerformance comparison:")
            print(f"  • Single-threaded: {single_time:.2f} seconds")
            print(f"  • Multiprocessing: {multi_time:.2f} seconds")
            print(f"  • Speedup: {single_time / multi_time:.2f}x")

            single_scores = results_single['normalized_scores']
            multi_scores = results_multi['normalized_scores']
            print("  • Results match: {}".format("✓" if single_scores == multi_scores else "✗"))
        else:
            print("One or both analyses failed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
