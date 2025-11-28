#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for LSTM Analysis functionality
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lstm_analysis import (
    ContinuousSequence, AnomalyResult, LSTMAnalyzer, 
    ContinuousSequenceDetector, LSTMTimeSeriesAnalyzer
)
from analysis import LogEntry


def create_test_log_entries():
    """Create test log entries for testing"""
    entries = []
    
    # Create continuous sequence for rank 0
    base_time = time.time()
    for i in range(20):
        entry = LogEntry(
            raw_line=f"test_line_{i}",
            save_count=1,
            timestamp=base_time + i * 0.1,  # Normal intervals
            rank=0,
            function=f"test_func_{i}",
            data_size=1024,
            stream="0x1234",
            op_count=i
        )
        entries.append(entry)
    
    # Add some anomalies (longer intervals)
    for i in range(20, 25):
        entry = LogEntry(
            raw_line=f"test_line_{i}",
            save_count=1,
            timestamp=base_time + i * 0.1 + 0.5,  # Longer interval
            rank=0,
            function=f"test_func_{i}",
            data_size=1024,
            stream="0x1234",
            op_count=i
        )
        entries.append(entry)
    
    # Create another sequence for rank 1
    for i in range(15):
        entry = LogEntry(
            raw_line=f"test_line_rank1_{i}",
            save_count=2,
            timestamp=base_time + i * 0.15,
            rank=1,
            function=f"test_func_rank1_{i}",
            data_size=2048,
            stream="0x5678",
            op_count=i
        )
        entries.append(entry)
    
    return entries


def test_continuous_sequence():
    """Test ContinuousSequence class"""
    print("Testing ContinuousSequence class...")
    
    seq = ContinuousSequence(0, "0x1234", 1, 1)
    
    # Add test entries
    base_time = time.time()
    for i in range(5):
        entry = LogEntry(
            raw_line=f"test_{i}",
            save_count=1,
            timestamp=base_time + i * 0.1,
            rank=0,
            function=f"func_{i}",
            data_size=1024,
            stream="0x1234",
            op_count=i
        )
        seq.add_entry(entry)
    
    print(f"  Sequence length: {seq.get_sequence_length()}")
    print(f"  Is continuous: {seq.is_continuous()}")
    print(f"  Time intervals: {seq.get_time_intervals()}")
    
    assert seq.get_sequence_length() == 5
    assert seq.is_continuous() == True
    assert len(seq.get_time_intervals()) == 4
    
    print("  ✓ ContinuousSequence test passed")


def test_anomaly_result():
    """Test AnomalyResult class"""
    print("Testing AnomalyResult class...")
    
    seq = ContinuousSequence(0, "0x1234", 1, 1)
    anomaly_indices = [2, 4]
    deviation_scores = [2.5, 3.1]
    predicted_values = [0.1, 0.1, 0.1, 0.1]
    actual_values = [0.1, 0.1, 0.3, 0.1]
    
    result = AnomalyResult(seq, anomaly_indices, deviation_scores, predicted_values, actual_values)
    summary = result.get_anomaly_summary()
    
    print(f"  Anomalies found: {summary['anomalies_found']}")
    print(f"  Max deviation: {summary['max_deviation']}")
    
    assert summary['anomalies_found'] == 2
    assert summary['max_deviation'] == 3.1
    
    print("  ✓ AnomalyResult test passed")


def test_continuous_sequence_detector():
    """Test ContinuousSequenceDetector class"""
    print("Testing ContinuousSequenceDetector class...")
    
    detector = ContinuousSequenceDetector()
    test_entries = create_test_log_entries()
    
    # Group by rank
    rank_entries = {}
    for entry in test_entries:
        if entry.rank not in rank_entries:
            rank_entries[entry.rank] = []
        rank_entries[entry.rank].append(entry)
    
    sequences = detector.detect_sequences(rank_entries)
    
    print(f"  Found {len(sequences)} sequences")
    
    # Check sequences
    rank0_sequences = [s for s in sequences if s.rank == 0]
    rank1_sequences = [s for s in sequences if s.rank == 1]
    
    print(f"  Rank 0 sequences: {len(rank0_sequences)}")
    print(f"  Rank 1 sequences: {len(rank1_sequences)}")
    
    assert len(sequences) > 0
    assert len(rank0_sequences) > 0
    assert len(rank1_sequences) > 0
    
    print("  ✓ ContinuousSequenceDetector test passed")


def test_lstm_analyzer():
    """Test LSTMAnalyzer class"""
    print("Testing LSTMAnalyzer class...")
    
    analyzer = LSTMAnalyzer(sequence_length=5, threshold=2.0)
    
    # Create a test sequence
    seq = ContinuousSequence(0, "0x1234", 1, 1)
    base_time = time.time()
    
    # Add entries with normal intervals
    for i in range(10):
        entry = LogEntry(
            raw_line=f"test_{i}",
            save_count=1,
            timestamp=base_time + i * 0.1,
            rank=0,
            function=f"func_{i}",
            data_size=1024,
            stream="0x1234",
            op_count=i
        )
        seq.add_entry(entry)
    
    # Analyze sequence
    result = analyzer.analyze_sequence(seq)
    
    print(f"  Analysis completed: {result.get_anomaly_summary()}")
    
    assert result.sequence == seq
    print("  ✓ LSTMAnalyzer test passed")


def test_lstm_time_series_analyzer():
    """Test LSTMTimeSeriesAnalyzer class"""
    print("Testing LSTMTimeSeriesAnalyzer class...")
    
    analyzer = LSTMTimeSeriesAnalyzer(sequence_length=5, threshold=2.0)
    test_entries = create_test_log_entries()
    
    # Group by rank
    rank_entries = {}
    for entry in test_entries:
        if entry.rank not in rank_entries:
            rank_entries[entry.rank] = []
        rank_entries[entry.rank].append(entry)
    
    # Analyze logs
    results = analyzer.analyze_logs(rank_entries)
    
    print(f"  Analysis results for {len(results)} ranks")
    
    # Generate report
    report = analyzer.generate_report()
    print("  Report generated successfully")
    
    # Check specific rank anomalies (now returns dict grouped by stream)
    rank0_anomalies = analyzer.get_rank_anomalies(0)
    total_anomalies = sum(len(stream_anomalies) for stream_anomalies in rank0_anomalies.values())
    print(f"  Rank 0 anomalies: {total_anomalies} across {len(rank0_anomalies)} streams")
    
    # Test specific stream anomalies
    if rank0_anomalies:
        first_stream = list(rank0_anomalies.keys())[0]
        stream_specific_anomalies = analyzer.get_rank_stream_anomalies(0, first_stream)
        print(f"  Stream {first_stream} anomalies: {len(stream_specific_anomalies)}")
    
    assert len(results) > 0
    print("  ✓ LSTMTimeSeriesAnalyzer test passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("LSTM Analysis Test Suite")
    print("=" * 60)
    
    try:
        test_continuous_sequence()
        test_anomaly_result()
        test_continuous_sequence_detector()
        test_lstm_analyzer()
        test_lstm_time_series_analyzer()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
