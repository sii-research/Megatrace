#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Script for LSTM Analysis
Run this script to quickly start LSTM analysis on your logs
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
    except ImportError:
        missing_deps.append("torch")
        print("✗ PyTorch not available - will use fallback methods")
    
    try:
        import sklearn
        print(f"✓ Scikit-learn available")
    except ImportError:
        missing_deps.append("scikit-learn")
        print("✗ Scikit-learn not available - will use basic scaling")
    
    try:
        import numpy
        print(f"✓ NumPy available")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ NumPy not available - required!")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas available")
    except ImportError:
        missing_deps.append("pandas")
        print("✗ Pandas not available - required!")
        return False
    
    if missing_deps:
        print(f"\nMissing optional dependencies: {missing_deps}")
        print("Install with: pip install " + " ".join(missing_deps))
        print("LSTM analysis will still work with reduced functionality")
    
    return True


def quick_analysis(log_path, sequence_length=10, threshold=2.0, verbose=False):
    """Perform quick LSTM analysis"""
    try:
        from lstm_analysis import LSTMTimeSeriesAnalyzer
        from analysis import DistributedLogAnalyzer
        
        print(f"\nStarting LSTM analysis...")
        print(f"Log path: {log_path}")
        print(f"Sequence length: {sequence_length}")
        print(f"Threshold: {threshold}")
        
        # Initialize analyzer
        lstm_analyzer = LSTMTimeSeriesAnalyzer(
            sequence_length=sequence_length,
            threshold=threshold
        )
        
        # Parse logs
        print("\nParsing log files...")
        log_analyzer = DistributedLogAnalyzer(log_path, verbose=verbose)
        log_analyzer.discover_log_files()
        rank_entries = log_analyzer.parse_log_files()
        
        if not rank_entries:
            print("✗ No log data found or parsing failed!")
            return False
        
        print(f"✓ Found data for {len(rank_entries)} ranks")
        
        # Perform analysis
        print("\nPerforming LSTM analysis...")
        results = lstm_analyzer.analyze_logs(rank_entries)
        
        # Generate report
        print("\nGenerating report...")
        report = lstm_analyzer.generate_report()
        print(report)
        
        # Save results to file
        output_file = "lstm_analysis_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✓ Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def interactive_mode():
    """Interactive mode for parameter selection"""
    print("\n" + "=" * 60)
    print("Interactive LSTM Analysis Setup")
    print("=" * 60)
    
    # Get log path
    while True:
        log_path = input("\nEnter log file/directory path: ").strip()
        if not log_path:
            print("Please enter a valid path")
            continue
        
        if os.path.exists(log_path):
            break
        else:
            print(f"Path not found: {log_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None, None, None, None
    
    # Get sequence length
    while True:
        try:
            seq_len = input("\nEnter sequence length (5-20, default 10): ").strip()
            if not seq_len:
                sequence_length = 10
                break
            sequence_length = int(seq_len)
            if 5 <= sequence_length <= 20:
                break
            else:
                print("Sequence length should be between 5 and 20")
        except ValueError:
            print("Please enter a valid number")
    
    # Get threshold
    while True:
        try:
            thresh = input("\nEnter anomaly threshold (0.5-5.0, default 2.0): ").strip()
            if not thresh:
                threshold = 2.0
                break
            threshold = float(thresh)
            if 0.5 <= threshold <= 5.0:
                break
            else:
                print("Threshold should be between 0.5 and 5.0")
        except ValueError:
            print("Please enter a valid number")
    
    # Get verbose mode
    verbose = input("\nEnable verbose output? (y/n, default n): ").strip().lower() == 'y'
    
    return log_path, sequence_length, threshold, verbose


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Quick Start LSTM Analysis for Distributed Training Logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis with default parameters
  python quick_start_lstm.py /path/to/logs
  
  # Custom parameters
  python quick_start_lstm.py /path/to/logs --sequence_length 15 --threshold 1.5
  
  # Interactive mode
  python quick_start_lstm.py --interactive
  
  # Verbose output
  python quick_start_lstm.py /path/to/logs --verbose
        """
    )
    
    parser.add_argument('log_path', nargs='?', help='Path to log files or directory')
    parser.add_argument('--sequence_length', type=int, default=10, 
                       help='LSTM sequence length (default: 10)')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Anomaly detection threshold (default: 2.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    print("LSTM Analysis Quick Start")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Critical dependencies missing. Please install required packages.")
        print("Run: pip install -r requirements.txt")
        return 1
    
    # Determine mode
    if args.interactive:
        log_path, sequence_length, threshold, verbose = interactive_mode()
        if log_path is None:
            print("Interactive setup cancelled.")
            return 0
    elif args.log_path:
        log_path = args.log_path
        sequence_length = args.sequence_length
        threshold = args.threshold
        verbose = args.verbose
    else:
        print("Error: Please provide log path or use --interactive mode")
        parser.print_help()
        return 1
    
    # Perform analysis
    success = quick_analysis(log_path, sequence_length, threshold, verbose)
    
    if success:
        print("\n" + "=" * 60)
        print("LSTM Analysis completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the analysis results above")
        print("2. Check the saved results file: lstm_analysis_results.txt")
        print("3. For more control, use: python lstm_analysis.py --help")
        print("4. For examples, run: python example_lstm_usage.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("LSTM Analysis failed!")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check that the log path is correct")
        print("2. Ensure log files are readable")
        print("3. Try with --verbose flag for more details")
        print("4. Check the LSTM_README.md for more information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
