#!/usr/bin/env python3
"""
Test script to verify IQR method with simulated slow nodes
"""

import os
import tempfile
import shutil
from parallel_slow_detector import ParallelSlowDetector

def create_test_logs_with_slow_nodes():
    """Create test log files with simulated slow nodes"""
    test_dir = "test_logs_iqr"
    
    # Clean up if exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Create config.yaml
    config_content = """TP: 4
PP: 2
world_size: 8
"""
    with open(os.path.join(test_dir, "config.yaml"), "w") as f:
        f.write(config_content)
    
    # Create log files with simulated slow nodes
    # Rank 2 will be intentionally slow
    base_time = 1000.0
    
    for rank in range(8):
        log_file = os.path.join(test_dir, f"rank_{rank}.log")
        with open(log_file, "w") as f:
            for op_count in range(10):
                # Rank 2 is slow (add 20 seconds delay to make it more obvious)
                if rank == 2:
                    timestamp = base_time + op_count * 10 + 20
                else:
                    timestamp = base_time + op_count * 10
                
                f.write(f"[save_count 1] [{timestamp}] [Rank {rank}] Fun AllReduce Data 1 stream 0x12345678 opCount {op_count}\n")
    
    return test_dir

def test_iqr_method():
    """Test IQR method with simulated slow nodes"""
    print("Testing IQR Method with Simulated Slow Nodes")
    print("=" * 50)
    
    # Create test logs
    test_dir = create_test_logs_with_slow_nodes()
    print(f"Created test logs in: {test_dir}")
    
    try:
        # Run analysis
        detector = ParallelSlowDetector(test_dir, config_path=os.path.join(test_dir, "config.yaml"), verbose=True, use_multiprocessing=False)
        results = detector.analyze_parallel_slow_nodes()
        
        if results:
            print("\n" + "=" * 50)
            print("ANALYSIS RESULTS")
            print("=" * 50)
            detector.print_results(results)
            
            # Check if Rank 2 was detected as slow
            normalized_scores = results['normalized_scores']
            if 2 in normalized_scores:
                rank2_scores = normalized_scores[2]
                total_score = sum(rank2_scores.values())
                print(f"\nRank 2 total score: {total_score}")
                if total_score > 0:
                    print("✓ IQR method successfully detected slow node (Rank 2)")
                else:
                    print("✗ IQR method did not detect slow node (Rank 2)")
            else:
                print("✗ Rank 2 not found in results")
        else:
            print("✗ Analysis failed")
            
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")

if __name__ == "__main__":
    test_iqr_method()
