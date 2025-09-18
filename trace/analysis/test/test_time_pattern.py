#!/usr/bin/env python3
"""
Test script for time pattern analysis functionality.
Note: For unified diagnostics, prefer: python analysis/test_analyzer.py --log-path <logs> --test-type hang --verbose
"""

import os
import sys
sys.path.insert(0, 'analysis')
from analysis import DistributedLogAnalyzer

def test_time_pattern_analysis(logs_path: str, verbose: bool = False):
    """
    Test time pattern analysis functionality
    
    Args:
        logs_path: Path to the logs directory
        verbose: Whether to enable verbose logging
    """
    print(f"\n{'='*60}")
    print("Testing Time Pattern Analysis Functionality")
    print(f"{'='*60}")
    print(f"Using specified log path: {logs_path}")
    
    # Check if log path exists
    if not os.path.exists(logs_path):
        print(f"Error: Log path '{logs_path}' does not exist!")
        return
    
    # Create analyzer instance
    analyzer = DistributedLogAnalyzer(logs_path, verbose=verbose)
    
    try:
        # Run complete analysis including time pattern analysis
        analyzer.run()
        
        # Get analysis results
        time_pattern_results = analyzer.analysis_results.get('time_pattern_analysis')
        
        if time_pattern_results:
            print(f"\nTime pattern analysis completed successfully!")
            print(f"Analysis Summary:")
            summary = time_pattern_results['summary']
            print(f"  • Total Patterns Analyzed: {summary['total_patterns']}")
            print(f"  • Total Anomalies Detected: {summary['total_anomalies']}")
            
            if summary['anomaly_types']:
                print(f"\nAnomaly Types Found:")
                for anomaly_type, count in summary['anomaly_types'].items():
                    print(f"  • {anomaly_type}: {count}")
            
            if summary['severity_distribution']:
                print(f"\nSeverity Distribution:")
                for severity, count in summary['severity_distribution'].items():
                    print(f"  • {severity}: {count}")
            
            # Show some example patterns
            if time_pattern_results['patterns']:
                print(f"\nExample Patterns (first 5):")
                pattern_count = 0
                for pattern_id, pattern in time_pattern_results['patterns'].items():
                    if pattern_count >= 5:
                        break
                    print(f"  • {pattern.api_name} (Rank {pattern.rank}, Stream {pattern.stream})")
                    print(f"    Type: {pattern.pattern_type}, Calls: {pattern.call_count}")
                    print(f"    Mean Interval: {pattern.mean_interval:.3f}s, Std: {pattern.std_interval:.3f}s")
                    pattern_count += 1
            
        else:
            print(f"\nTime pattern analysis not available or failed")
            print(f"   This could be due to missing dependencies (numpy, pandas)")
            print(f"   or insufficient log data for pattern analysis")
        
        print(f"\nAnalysis completed!")
        
    except Exception as e:
        print(f"Error during time pattern analysis: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Time Pattern Analysis')
    parser.add_argument('--log-path', required=True, help='Path to the logs directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    test_time_pattern_analysis(args.log_path, args.verbose)

if __name__ == "__main__":
    main()
