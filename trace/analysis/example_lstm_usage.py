#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating LSTM analysis usage
IMPORTANT: Each rank's logs are grouped by stream for LSTM analysis
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Basic usage example - demonstrates stream grouping"""
    print("=" * 60)
    print("Basic LSTM Analysis Usage Example")
    print("IMPORTANT: Logs are grouped by rank AND stream for analysis")
    print("=" * 60)
    
    try:
        from lstm_analysis import LSTMTimeSeriesAnalyzer
        from analysis import DistributedLogAnalyzer
        
        # Initialize LSTM analyzer
        lstm_analyzer = LSTMTimeSeriesAnalyzer(
            sequence_length=8,    # Use 8 operations for prediction
            threshold=1.5         # Lower threshold for more sensitive detection
        )
        
        print("✓ LSTM analyzer initialized successfully")
        print(f"  Sequence length: {lstm_analyzer.sequence_detector.sequences}")
        print(f"  Threshold: {lstm_analyzer.lstm_analyzer.threshold}")
        print("  Note: Analysis will be performed separately for each rank's streams")
        
        # Example: analyze logs from a specific path
        log_path = "./logs"  # Adjust this path as needed
        
        if os.path.exists(log_path):
            print(f"\nAnalyzing logs from: {log_path}")
            
            # Initialize log analyzer
            log_analyzer = DistributedLogAnalyzer(log_path, verbose=False)
            log_analyzer.discover_log_files()
            rank_entries = log_analyzer.parse_log_files()
            
            if rank_entries:
                print(f"✓ Found log data for {len(rank_entries)} ranks")
                
                # Perform LSTM analysis (automatically grouped by rank and stream)
                print("\nPerforming LSTM analysis (grouped by rank and stream)...")
                results = lstm_analyzer.analyze_logs(rank_entries)
                
                # Generate report
                report = lstm_analyzer.generate_report()
                print("\n" + report)
                
                # Get specific rank anomalies (grouped by stream)
                for rank in sorted(results.keys()):
                    print(f"\nRank {rank} anomalies by stream:")
                    stream_anomalies = lstm_analyzer.get_rank_anomalies(rank)
                    
                    if stream_anomalies:
                        for stream, anomalies in stream_anomalies.items():
                            print(f"  Stream {stream}: {len(anomalies)} sequences with anomalies")
                            for i, anomaly in enumerate(anomalies):
                                seq_info = anomaly['sequence_info']
                                anomaly_info = anomaly['anomalies']
                                print(f"    Sequence {i+1}: {seq_info['operations_count']} operations")
                                print(f"      Save count range: {seq_info['start_save_count']}-{seq_info['end_save_count']}")
                                print(f"      Anomalies: {anomaly_info['anomalies_found']}")
                                print(f"      Max deviation: {anomaly_info['max_deviation']:.3f}")
                    else:
                        print("    No anomalies detected in any stream")
            else:
                print("⚠ No log data found or parsing failed")
        else:
            print(f"⚠ Log path not found: {log_path}")
            print("  Please adjust the log_path variable in this script")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def example_stream_specific_analysis():
    """Example of stream-specific analysis"""
    print("\n" + "=" * 60)
    print("Stream-Specific LSTM Analysis Example")
    print("Each stream is analyzed independently for better accuracy")
    print("=" * 60)
    
    try:
        from lstm_analysis import LSTMTimeSeriesAnalyzer, ContinuousSequence, LogEntry
        
        # Create custom analyzer with different parameters
        custom_analyzer = LSTMTimeSeriesAnalyzer(
            sequence_length=12,   # Longer sequence for better pattern recognition
            threshold=2.5         # Higher threshold for fewer false positives
        )
        
        print("✓ Custom analyzer initialized")
        print(f"  Sequence length: {custom_analyzer.lstm_analyzer.sequence_length}")
        print(f"  Threshold: {custom_analyzer.lstm_analyzer.threshold}")
        print("  Note: Each stream will be analyzed with these parameters")
        
        # Create synthetic test data with multiple streams
        print("\nCreating synthetic test data with multiple streams...")
        test_sequences = create_synthetic_data_with_streams()
        
        # Analyze synthetic data
        results = custom_analyzer.analyze_logs(test_sequences)
        
        print(f"✓ Analysis completed for {len(results)} ranks")
        
        # Show results grouped by stream
        for rank, rank_results in results.items():
            print(f"\nRank {rank} results by stream:")
            
            # Group results by stream
            stream_results = {}
            for result in rank_results:
                stream = result.sequence.stream
                if stream not in stream_results:
                    stream_results[stream] = []
                stream_results[stream].append(result)
            
            for stream, stream_sequences in stream_results.items():
                print(f"  Stream {stream}: {len(stream_sequences)} sequences")
                
                for i, result in enumerate(stream_sequences):
                    summary = result.get_anomaly_summary()
                    print(f"    Sequence {i+1}: {summary['anomalies_found']} anomalies")
                    if summary['anomalies_found'] > 0:
                        print(f"      Max deviation: {summary['max_deviation']:.3f}")
                        print(f"      Anomaly positions: {summary['anomaly_indices']}")
        
    except Exception as e:
        print(f"✗ Error in stream-specific analysis: {e}")


def create_synthetic_data_with_streams():
    """Create synthetic log data with multiple streams for testing"""
    from collections import defaultdict
    
    # Create test entries with multiple streams
    entries = []
    base_time = time.time()
    
    # Rank 0: Stream 0x1234 (normal operations)
    for i in range(20):
        entry = LogEntry(
            raw_line=f"synth_line_{i}",
            save_count=1,
            timestamp=base_time + i * 0.1,  # Normal intervals
            rank=0,
            function=f"synth_func_{i}",
            data_size=1024,
            stream="0x1234",
            op_count=i
        )
        entries.append(entry)
    
    # Rank 0: Stream 0x5678 (different pattern, some anomalies)
    for i in range(15):
        interval = 0.15
        if i in [5, 10]:  # Anomalies
            interval = 0.6  # 4x longer
        entry = LogEntry(
            raw_line=f"synth_line_stream2_{i}",
            save_count=1,
            timestamp=base_time + i * interval,
            rank=0,
            function=f"synth_func_stream2_{i}",
            data_size=2048,
            stream="0x5678",
            op_count=i
        )
        entries.append(entry)
    
    # Rank 1: Stream 0x9ABC (fast operations)
    for i in range(25):
        entry = LogEntry(
            raw_line=f"synth_line_rank1_{i}",
            save_count=2,
            timestamp=base_time + i * 0.05,  # Fast operations
            rank=1,
            function=f"synth_func_rank1_{i}",
            data_size=512,
            stream="0x9ABC",
            op_count=i
        )
        entries.append(entry)
    
    # Rank 1: Stream 0xDEF0 (slow operations with anomalies)
    for i in range(18):
        interval = 0.3
        if i in [8, 15]:  # Anomalies
            interval = 1.2  # 4x longer
        entry = LogEntry(
            raw_line=f"synth_line_rank1_slow_{i}",
            save_count=2,
            timestamp=base_time + i * interval,
            rank=1,
            function=f"synth_func_rank1_slow_{i}",
            data_size=4096,
            stream="0xDEF0",
            op_count=i
        )
        entries.append(entry)
    
    # Group by rank
    rank_entries = defaultdict(list)
    for entry in entries:
        rank_entries[entry.rank].append(entry)
    
    return rank_entries


def example_parameter_tuning():
    """Example of parameter tuning for different scenarios"""
    print("\n" + "=" * 60)
    print("Parameter Tuning Examples")
    print("Note: Parameters apply to each stream independently")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'High Sensitivity (More Anomalies)',
            'sequence_length': 5,
            'threshold': 1.0,
            'description': 'Shorter sequences, lower threshold for detecting subtle anomalies in each stream'
        },
        {
            'name': 'Balanced (Default)',
            'sequence_length': 10,
            'threshold': 2.0,
            'description': 'Standard settings for general use across all streams'
        },
        {
            'name': 'High Precision (Fewer False Positives)',
            'sequence_length': 15,
            'threshold': 3.0,
            'description': 'Longer sequences, higher threshold for reliable detection per stream'
        },
        {
            'name': 'Performance Optimized',
            'sequence_length': 8,
            'threshold': 2.5,
            'description': 'Balanced performance and accuracy for each stream'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Sequence Length: {scenario['sequence_length']}")
        print(f"  Threshold: {scenario['threshold']}")
        print(f"  Description: {scenario['description']}")
        
        # Show how to create analyzer with these parameters
        print(f"  Usage:")
        print(f"    analyzer = LSTMTimeSeriesAnalyzer(")
        print(f"        sequence_length={scenario['sequence_length']},")
        print(f"        threshold={scenario['threshold']}")
        print(f"    )")
        print(f"    # These parameters will apply to each stream independently")


def example_stream_analysis_benefits():
    """Example demonstrating benefits of stream-based analysis"""
    print("\n" + "=" * 60)
    print("Benefits of Stream-Based LSTM Analysis")
    print("=" * 60)
    
    benefits = [
        {
            'title': '1. Temporal Correlation',
            'description': 'Operations within the same stream have similar execution patterns',
            'example': 'CUDA kernel launches in stream 0x1234 follow similar timing patterns',
            'benefit': 'Better prediction accuracy and anomaly detection'
        },
        {
            'title': '2. Stream-Specific Patterns',
            'description': 'Different streams may have different performance characteristics',
            'example': 'Stream 0x5678 (data transfer) vs Stream 0x9ABC (computation)',
            'benefit': 'Identify stream-specific performance issues'
        },
        {
            'title': '3. Precise Anomaly Localization',
            'description': 'Anomalies are reported per stream, not mixed across streams',
            'example': 'Anomaly in stream 0x1234 at operation 15 with deviation 2.5',
            'benefit': 'Easier debugging and performance optimization'
        },
        {
            'title': '4. Independent Modeling',
            'description': 'Each stream gets its own LSTM model',
            'example': 'Stream 0x1234 model learns from its own timing patterns',
            'benefit': 'No interference between different execution contexts'
        }
    ]
    
    for benefit in benefits:
        print(f"\n{benefit['title']}")
        print(f"  Description: {benefit['description']}")
        print(f"  Example: {benefit['example']}")
        print(f"  Benefit: {benefit['benefit']}")


def main():
    """Main function"""
    print("LSTM Analysis Examples")
    print("=" * 60)
    print("IMPORTANT: This system analyzes logs by grouping each rank's data by stream")
    print("Each stream is analyzed independently for better accuracy and relevance")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_stream_specific_analysis()
        example_parameter_tuning()
        example_stream_analysis_benefits()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        print("\nKey Points:")
        print("1. Each rank's logs are automatically grouped by stream")
        print("2. LSTM analysis is performed separately for each stream")
        print("3. This ensures temporal correlation and better accuracy")
        print("4. Anomalies are reported per stream for precise localization")
        print("\nTo run LSTM analysis on your logs:")
        print("1. Adjust the log_path in example_basic_usage()")
        print("2. Run: python example_lstm_usage.py")
        print("3. Or use directly: python lstm_analysis.py --log_path /path/to/logs")
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
