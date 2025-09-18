#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-threaded LSTM Time Series Analysis for Distributed Training Logs
- Analyze continuous opCount sequences for each rank using multiple threads
- Detect anomalies using LSTM prediction
- Calculate deviation scores for abnormal execution patterns
- Parallel processing for improved performance
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from collections import defaultdict
import logging
from pathlib import Path
import warnings
import threading
import concurrent.futures
import time
from queue import Queue
warnings.filterwarnings('ignore')

# Try to import LSTM-related libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. LSTM analysis will use simplified methods.")

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not available. Using basic scaling methods.")

# Import from local analysis module
try:
    from analysis import LogEntry, DistributedLogAnalyzer
except ImportError:
    # Fallback if import fails
    class LogEntry:
        def __init__(self, raw_line: str, save_count: int, timestamp: float, 
                     rank: int, function: str, data_size: int, stream: str, op_count: int):
            self.raw_line = raw_line
            self.save_count = save_count
            self.timestamp = timestamp
            self.rank = rank
            self.function = function
            self.data_size = data_size
            self.stream = stream
            self.op_count = int(op_count)


class ContinuousSequence:
    """Represents a continuous sequence of operations for LSTM analysis"""
    
    def __init__(self, rank: int, stream: str, start_save_count: int, end_save_count: int):
        self.rank = rank
        self.stream = stream
        self.start_save_count = start_save_count
        self.end_save_count = end_save_count
        self.entries: List[LogEntry] = []
        self.timestamps: List[float] = []
        self.op_counts: List[int] = []
        self.functions: List[str] = []
        
    def add_entry(self, entry: LogEntry):
        """Add a log entry to the sequence"""
        self.entries.append(entry)
        self.timestamps.append(entry.timestamp)
        self.op_counts.append(entry.op_count)
        self.functions.append(entry.function)
        
    def is_continuous(self) -> bool:
        """Check if opCounts are continuous"""
        if len(self.op_counts) < 2:
            return True
        
        for i in range(1, len(self.op_counts)):
            if self.op_counts[i] != self.op_counts[i-1] + 1:
                return False
        return True
    
    def get_sequence_length(self) -> int:
        """Get the length of the continuous sequence"""
        return len(self.entries)
    
    def get_time_intervals(self) -> List[float]:
        """Get time intervals between consecutive operations"""
        if len(self.timestamps) < 2:
            return []
        
        intervals = []
        for i in range(1, len(self.timestamps)):
            intervals.append(self.timestamps[i] - self.timestamps[i-1])
        return intervals


class AnomalyResult:
    """Result of anomaly detection for a sequence"""
    
    def __init__(self, sequence: ContinuousSequence, anomaly_indices: List[int], 
                 deviation_scores: List[float], predicted_values: List[float], 
                 actual_values: List[float]):
        self.sequence = sequence
        self.anomaly_indices = anomaly_indices
        self.deviation_scores = deviation_scores
        self.predicted_values = predicted_values
        self.actual_values = actual_values
        
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomalies"""
        if not self.anomaly_indices:
            return {"anomalies_found": 0, "max_deviation": 0.0}
        
        return {
            "anomalies_found": len(self.anomaly_indices),
            "max_deviation": max(self.deviation_scores),
            "anomaly_indices": self.anomaly_indices,
            "deviation_scores": self.deviation_scores
        }


class LSTMDataset(Dataset):
    """Dataset for LSTM training"""
    
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTMAnalyzer:
    """LSTM-based analyzer for distributed training logs"""
    
    def __init__(self, sequence_length: int = 10, hidden_size: int = 64, 
                 num_layers: int = 2, threshold: float = 2.0):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize scaler
        if SKLEARN_AVAILABLE:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        # Initialize model
        if TORCH_AVAILABLE:
            self.model = LSTMModel(1, hidden_size, num_layers, 1).to(self.device)
        else:
            self.model = None
            
    def _simple_scale(self, data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Simple scaling without sklearn"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return data, min_val, max_val
        
        scaled = (data - min_val) / (max_val - min_val)
        return scaled, min_val, max_val
    
    def _simple_inverse_scale(self, scaled_data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Simple inverse scaling without sklearn"""
        return scaled_data * (max_val - min_val) + min_val
    
    def _prepare_data(self, time_intervals: List[float]) -> Tuple[np.ndarray, float, float]:
        """Prepare data for LSTM analysis"""
        if not time_intervals:
            return np.array([]), 0.0, 1.0
            
        data = np.array(time_intervals).reshape(-1, 1)
        
        if self.scaler:
            data_scaled = self.scaler.fit_transform(data)
            return data_scaled, 0.0, 1.0  # scaler handles min/max
        else:
            return self._simple_scale(data)
    
    def _train_lstm(self, data: np.ndarray, epochs: int = 100) -> List[float]:
        """Train LSTM model and return predictions"""
        if not TORCH_AVAILABLE or len(data) <= self.sequence_length:
            return self._fallback_prediction(data)
        
        # Prepare dataset
        dataset = LSTMDataset(data, self.sequence_length)
        if len(dataset) == 0:
            return self._fallback_prediction(data)
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0 and total_loss < 0.001:
                break
        
        # Generate predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(data) - self.sequence_length):
                x = torch.FloatTensor(data[i:i + self.sequence_length]).unsqueeze(0).to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu().numpy()[0][0])
        
        return predictions
    
    def _fallback_prediction(self, data: np.ndarray) -> List[float]:
        """Fallback prediction method when LSTM is not available"""
        if len(data) < 2:
            return []
        
        # Simple moving average prediction
        predictions = []
        for i in range(1, len(data)):
            if i < 3:
                # Use simple average for first few points
                pred = np.mean(data[:i])
            else:
                # Use moving average
                pred = np.mean(data[i-3:i])
            predictions.append(pred)
        
        return predictions
    
    def _detect_anomalies(self, actual: List[float], predicted: List[float]) -> Tuple[List[int], List[float]]:
        """Detect anomalies based on prediction errors"""
        if len(actual) != len(predicted):
            return [], []
        
        anomalies = []
        deviation_scores = []
        
        for i, (act, pred) in enumerate(zip(actual, predicted)):
            if act == 0:  # Avoid division by zero
                deviation = abs(act - pred)
            else:
                deviation = abs(act - pred) / act
            
            deviation_scores.append(deviation)
            
            if deviation > self.threshold:
                anomalies.append(i)
        
        return anomalies, deviation_scores
    
    def analyze_sequence(self, sequence: ContinuousSequence) -> AnomalyResult:
        """Analyze a continuous sequence using LSTM"""
        if sequence.get_sequence_length() < self.sequence_length + 1:
            # Sequence too short for LSTM analysis
            return AnomalyResult(sequence, [], [], [], [])
        
        # Get time intervals
        time_intervals = sequence.get_time_intervals()
        if not time_intervals:
            return AnomalyResult(sequence, [], [], [], [])
        
        # Prepare data
        data_scaled, min_val, max_val = self._prepare_data(time_intervals)
        
        # Train LSTM and get predictions
        predictions_scaled = self._train_lstm(data_scaled)
        
        if not predictions_scaled:
            return AnomalyResult(sequence, [], [], [], [])
        
        # Inverse scale predictions
        if self.scaler:
            predictions = self.scaler.inverse_transform(
                np.array(predictions_scaled).reshape(-1, 1)
            ).flatten()
        else:
            predictions = self._simple_inverse_scale(
                np.array(predictions_scaled), min_val, max_val
            )
        
        # Detect anomalies
        anomaly_indices, deviation_scores = self._detect_anomalies(
            time_intervals[self.sequence_length:], predictions
        )
        
        return AnomalyResult(
            sequence, anomaly_indices, deviation_scores, 
            predictions.tolist(), time_intervals[self.sequence_length:]
        )


class ContinuousSequenceDetector:
    """Detector for finding continuous sequences in log data"""
    
    def __init__(self):
        self.sequences: List[ContinuousSequence] = []
    
    def detect_sequences(self, rank_entries: Dict[int, List[LogEntry]]) -> List[ContinuousSequence]:
        """Detect continuous sequences for each rank and stream"""
        self.sequences = []
        
        for rank, entries in rank_entries.items():
            # Group by stream
            stream_entries = defaultdict(list)
            for entry in entries:
                stream_entries[entry.stream].append(entry)
            
            # Find continuous sequences for each stream
            for stream, stream_entries_list in stream_entries.items():
                self._find_continuous_sequences(rank, stream, stream_entries_list)
        
        return self.sequences
    
    def _find_continuous_sequences(self, rank: int, stream: str, entries: List[LogEntry]):
        """Find continuous sequences within a stream"""
        if not entries:
            return
        
        # Sort by save_count and op_count
        sorted_entries = sorted(entries, key=lambda x: (x.save_count, x.op_count))
        
        current_sequence = None
        
        for entry in sorted_entries:
            if current_sequence is None:
                # Start new sequence
                current_sequence = ContinuousSequence(
                    rank, stream, entry.save_count, entry.save_count
                )
                current_sequence.add_entry(entry)
            else:
                # Check if this entry continues the sequence
                if (entry.save_count == current_sequence.end_save_count and 
                    entry.op_count == current_sequence.entries[-1].op_count + 1):
                    # Continue sequence
                    current_sequence.end_save_count = entry.save_count
                    current_sequence.add_entry(entry)
                else:
                    # End current sequence and start new one
                    if current_sequence.get_sequence_length() > 1:
                        self.sequences.append(current_sequence)
                    
                    current_sequence = ContinuousSequence(
                        rank, stream, entry.save_count, entry.save_count
                    )
                    current_sequence.add_entry(entry)
        
        # Add final sequence if it has multiple entries
        if current_sequence and current_sequence.get_sequence_length() > 1:
            self.sequences.append(current_sequence)


class MultiThreadedLSTMAnalyzer:
    """Multi-threaded LSTM analyzer for parallel processing"""
    
    def __init__(self, sequence_length: int = 10, hidden_size: int = 64, 
                 num_layers: int = 2, threshold: float = 2.0, max_workers: int = None):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.threshold = threshold
        self.max_workers = max_workers
        
        # Thread-safe results storage
        self.results_lock = threading.Lock()
        self.analysis_results: Dict[int, List[AnomalyResult]] = defaultdict(list)
        
    def analyze_rank_sequences(self, rank: int, sequences: List[ContinuousSequence]) -> List[AnomalyResult]:
        """Analyze sequences for a specific rank using LSTM"""
        rank_results = []
        
        for sequence in sequences:
            # Create a new LSTM analyzer instance for each sequence to avoid conflicts
            lstm_analyzer = LSTMAnalyzer(
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                threshold=self.threshold
            )
            
            result = lstm_analyzer.analyze_sequence(sequence)
            rank_results.append(result)
        
        return rank_results
    
    def analyze_logs_parallel(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[int, List[AnomalyResult]]:
        """Analyze logs using multi-threaded LSTM analysis"""
        print("Detecting continuous sequences...")
        
        # Detect sequences first
        detector = ContinuousSequenceDetector()
        sequences = detector.detect_sequences(rank_entries)
        
        print(f"Found {len(sequences)} continuous sequences")
        
        # Group sequences by rank
        rank_sequences = defaultdict(list)
        for seq in sequences:
            rank_sequences[seq.rank].append(seq)
        
        # Group sequences by rank and stream for better organization
        rank_stream_sequences = defaultdict(lambda: defaultdict(list))
        for seq in sequences:
            rank_stream_sequences[seq.rank][seq.stream].append(seq)
        
        print(f"Processing {len(rank_sequences)} ranks with multi-threading...")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit analysis tasks for each rank
            future_to_rank = {}
            for rank, rank_seqs in rank_sequences.items():
                future = executor.submit(self.analyze_rank_sequences, rank, rank_seqs)
                future_to_rank[future] = rank
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_rank):
                rank = future_to_rank[future]
                try:
                    rank_results = future.result()
                    
                    # Store results thread-safely
                    with self.results_lock:
                        self.analysis_results[rank] = rank_results
                    
                    # Print progress
                    total_sequences = len(rank_results)
                    anomalies_found = sum(len(r.anomaly_indices) for r in rank_results)
                    print(f"✓ Rank {rank} completed: {total_sequences} sequences, {anomalies_found} anomalies")
                    
                except Exception as e:
                    print(f"✗ Error analyzing Rank {rank}: {e}")
                    with self.results_lock:
                        self.analysis_results[rank] = []
        
        return self.analysis_results


class LSTMTimeSeriesAnalyzer:
    """Main analyzer class for LSTM-based time series analysis with multi-threading support"""
    
    def __init__(self, sequence_length: int = 10, threshold: float = 2.0, 
                 max_workers: int = None, use_multithreading: bool = True):
        self.sequence_detector = ContinuousSequenceDetector()
        self.lstm_analyzer = LSTMAnalyzer(
            sequence_length=sequence_length,
            threshold=threshold
        )
        self.analysis_results: Dict[int, List[AnomalyResult]] = defaultdict(list)
        
        # Multi-threading support
        self.use_multithreading = use_multithreading
        if use_multithreading:
            self.multi_threaded_analyzer = MultiThreadedLSTMAnalyzer(
                sequence_length=sequence_length,
                threshold=threshold,
                max_workers=max_workers
            )
    
    def analyze_logs(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[int, List[AnomalyResult]]:
        """Analyze logs using LSTM for each rank, with optional multi-threading"""
        if self.use_multithreading:
            print("Using multi-threaded analysis for improved performance...")
            return self.multi_threaded_analyzer.analyze_logs_parallel(rank_entries)
        else:
            print("Using single-threaded analysis...")
            return self._analyze_logs_single_threaded(rank_entries)
    
    def _analyze_logs_single_threaded(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[int, List[AnomalyResult]]:
        """Single-threaded analysis (original implementation)"""
        print("Detecting continuous sequences...")
        sequences = self.sequence_detector.detect_sequences(rank_entries)
        
        print(f"Found {len(sequences)} continuous sequences")
        
        # Group sequences by rank and stream
        rank_stream_sequences = defaultdict(lambda: defaultdict(list))
        for seq in sequences:
            rank_stream_sequences[seq.rank][seq.stream].append(seq)
        
        # Analyze each rank's sequences grouped by stream
        for rank in sorted(rank_stream_sequences.keys()):
            stream_sequences = rank_stream_sequences[rank]
            total_sequences = sum(len(seqs) for seqs in stream_sequences.values())
            print(f"\nAnalyzing Rank {rank} with {len(stream_sequences)} streams, {total_sequences} total sequences...")
            
            for stream in sorted(stream_sequences.keys()):
                sequences_in_stream = stream_sequences[stream]
                print(f"  Stream {stream}: {len(sequences_in_stream)} sequences")
                
                for i, sequence in enumerate(sequences_in_stream):
                    print(f"    Sequence {i+1}: {sequence.get_sequence_length()} operations "
                          f"(save_count {sequence.start_save_count}-{sequence.end_save_count})")
                    
                    result = self.lstm_analyzer.analyze_sequence(sequence)
                    self.analysis_results[rank].append(result)
                    
                    # Print anomaly summary
                    summary = result.get_anomaly_summary()
                    if summary["anomalies_found"] > 0:
                        print(f"      Found {summary['anomalies_found']} anomalies "
                              f"(max deviation: {summary['max_deviation']:.3f})")
                    else:
                        print("      No anomalies detected")
        
        return self.analysis_results
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report grouped by rank and stream"""
        report = []
        report.append("=" * 80)
        report.append("LSTM Time Series Analysis Report (Grouped by Rank and Stream)")
        if self.use_multithreading:
            report.append("Multi-threaded analysis enabled for improved performance")
        report.append("=" * 80)
        
        total_anomalies = 0
        total_sequences = 0
        total_streams = 0
        
        for rank in sorted(self.analysis_results.keys()):
            rank_results = self.analysis_results[rank]
            rank_anomalies = sum(len(r.anomaly_indices) for r in rank_results)
            total_anomalies += rank_anomalies
            total_sequences += len(rank_results)
            
            # Group results by stream
            stream_results = defaultdict(list)
            for result in rank_results:
                stream_results[result.sequence.stream].append(result)
            
            total_streams += len(stream_results)
            
            report.append(f"\nRank {rank}:")
            report.append(f"  Total Streams: {len(stream_results)}")
            report.append(f"  Total Sequences: {len(rank_results)}")
            report.append(f"  Total Anomalies: {rank_anomalies}")
            
            # Report by stream
            for stream in sorted(stream_results.keys()):
                stream_sequences = stream_results[stream]
                stream_anomalies = sum(len(r.anomaly_indices) for r in stream_sequences)
                
                report.append(f"\n    Stream {stream}:")
                report.append(f"      Sequences: {len(stream_sequences)}")
                report.append(f"      Anomalies: {stream_anomalies}")
                
                for i, result in enumerate(stream_sequences):
                    summary = result.get_anomaly_summary()
                    report.append(f"        Sequence {i+1}: {result.sequence.get_sequence_length()} operations")
                    report.append(f"          Save Count Range: {result.sequence.start_save_count}-{result.sequence.end_save_count}")
                    report.append(f"          Anomalies Found: {summary['anomalies_found']}")
                    
                    if summary['anomalies_found'] > 0:
                        report.append(f"          Max Deviation: {summary['max_deviation']:.3f}")
                        report.append(f"          Anomaly Indices: {summary['anomaly_indices']}")
                        report.append(f"          Deviation Scores: {[f'{score:.3f}' for score in summary['deviation_scores']]}")
        
        report.append(f"\n" + "=" * 80)
        report.append(f"Summary:")
        report.append(f"  Total Ranks: {len(self.analysis_results)}")
        report.append(f"  Total Streams: {total_streams}")
        report.append(f"  Total Sequences: {total_sequences}")
        report.append(f"  Total Anomalies: {total_anomalies}")
        if self.use_multithreading:
            report.append(f"  Analysis Method: Multi-threaded")
        else:
            report.append(f"  Analysis Method: Single-threaded")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_rank_anomalies(self, rank: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed anomaly information for a specific rank, grouped by stream"""
        if rank not in self.analysis_results:
            return {}
        
        stream_anomalies = defaultdict(list)
        for result in self.analysis_results[rank]:
            summary = result.get_anomaly_summary()
            if summary['anomalies_found'] > 0:
                stream_anomalies[result.sequence.stream].append({
                    'sequence_info': {
                        'start_save_count': result.sequence.start_save_count,
                        'end_save_count': result.sequence.end_save_count,
                        'stream': result.sequence.stream,
                        'operations_count': result.sequence.get_sequence_length()
                    },
                    'anomalies': summary
                })
        
        return dict(stream_anomalies)
    
    def get_rank_stream_anomalies(self, rank: int, stream: str) -> List[Dict[str, Any]]:
        """Get detailed anomaly information for a specific rank and stream"""
        if rank not in self.analysis_results:
            return []
        
        anomalies = []
        for result in self.analysis_results[rank]:
            if result.sequence.stream == stream:
                summary = result.get_anomaly_summary()
                if summary['anomalies_found'] > 0:
                    anomalies.append({
                        'sequence_info': {
                            'start_save_count': result.sequence.start_save_count,
                            'end_save_count': result.sequence.end_save_count,
                            'stream': result.sequence.stream,
                            'operations_count': result.sequence.get_sequence_length()
                        },
                        'anomalies': summary
                    })
        
        return anomalies


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-threaded LSTM Time Series Analysis for Distributed Training Logs')
    parser.add_argument('--log_path', required=True, help='Path to log files or directory')
    parser.add_argument('--sequence_length', type=int, default=10, help='LSTM sequence length')
    parser.add_argument('--threshold', type=float, default=2.0, help='Anomaly detection threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no_multithreading', action='store_true', help='Disable multi-threading')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker threads')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LSTMTimeSeriesAnalyzer(
        sequence_length=args.sequence_length,
        threshold=args.threshold,
        use_multithreading=not args.no_multithreading,
        max_workers=args.max_workers
    )
    
    # Parse logs (assuming we have access to DistributedLogAnalyzer)
    try:
        from analysis import DistributedLogAnalyzer
        log_analyzer = DistributedLogAnalyzer(args.log_path, args.verbose)
        log_analyzer.discover_log_files()
        rank_entries = log_analyzer.parse_log_files()
        
        if not rank_entries:
            print("No log data found or parsing failed!")
            return
        
        # Perform LSTM analysis
        start_time = time.time()
        results = analyzer.analyze_logs(rank_entries)
        end_time = time.time()
        
        # Generate and print report
        report = analyzer.generate_report()
        print(report)
        
        # Print performance information
        if analyzer.use_multithreading:
            print(f"\nPerformance: Analysis completed in {end_time - start_time:.2f} seconds")
            print(f"Multi-threading enabled for improved performance")
        
    except ImportError:
        print("Error: Could not import DistributedLogAnalyzer")
        print("Please ensure the analysis module is available")
        return


if __name__ == "__main__":
    main()
