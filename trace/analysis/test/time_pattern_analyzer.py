#!/usr/bin/env python3
"""
Time Pattern Analyzer for Distributed Training Logs
Analyzes time patterns of API calls across ranks and streams
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class TimePattern:
    """Time pattern information for an API call"""
    api_name: str
    rank: int
    stream: str
    mean_interval: float
    std_interval: float
    min_interval: float
    max_interval: float
    p50_interval: float
    p95_interval: float
    p99_interval: float
    call_count: int
    total_duration: float
    pattern_type: str  # 'regular', 'irregular', 'periodic', 'trending'

@dataclass
class TimeAnomaly:
    """Time anomaly detection result"""
    api_name: str
    rank: int
    stream: str
    anomaly_type: str  # 'interval_outlier', 'missing_call', 'unexpected_call', 'timing_drift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: float
    expected_interval: float
    actual_interval: float
    z_score: float

class TimePatternAnalyzer:
    """Analyzes time patterns in distributed training logs"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.patterns = {}  # Store detected patterns
        self.anomalies = []  # Store detected anomalies
        
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.ERROR)
        return logging.getLogger(__name__)
    
    def analyze_time_patterns(self, log_entries: Dict[int, List]) -> Dict[str, Any]:
        """
        Analyze time patterns from log entries
        
        Args:
            log_entries: Dictionary of rank -> list of log entries
            
        Returns:
            Analysis results with patterns and anomalies
        """
        self.logger.info("Starting time pattern analysis...")
        
        # Extract time series data for each API call
        api_time_series = self._extract_time_series(log_entries)
        
        # Analyze patterns for each API
        for api_name, rank_stream_data in api_time_series.items():
            for (rank, stream), timestamps in rank_stream_data.items():
                if len(timestamps) < 2:
                    continue
                    
                # Calculate time intervals
                intervals = self._calculate_intervals(timestamps)
                
                # Detect pattern
                pattern = self._detect_pattern(api_name, rank, stream, intervals, timestamps)
                self.patterns[f"{api_name}_{rank}_{stream}"] = pattern
                
                # Detect anomalies
                anomalies = self._detect_anomalies(api_name, rank, stream, intervals, timestamps, pattern)
                self.anomalies.extend(anomalies)
        
        return {
            'patterns': self.patterns,
            'anomalies': self.anomalies,
            'summary': self._generate_summary()
        }
    
    def _extract_time_series(self, log_entries: Dict[int, List]) -> Dict[str, Dict[Tuple[int, str], List[float]]]:
        """Extract time series data for each API call"""
        api_time_series = defaultdict(lambda: defaultdict(list))
        
        for rank, entries in log_entries.items():
            for entry in entries:
                api_name = entry.function
                stream = entry.stream
                timestamp = entry.timestamp
                
                api_time_series[api_name][(rank, stream)].append(timestamp)
        
        # Sort timestamps for each API/rank/stream combination
        for api_name in api_time_series:
            for (rank, stream) in api_time_series[api_name]:
                api_time_series[api_name][(rank, stream)].sort()
        
        return api_time_series
    
    def _calculate_intervals(self, timestamps: List[float]) -> List[float]:
        """Calculate time intervals between consecutive calls"""
        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            intervals.append(interval)
        return intervals
    
    def _detect_pattern(self, api_name: str, rank: int, stream: str, 
                       intervals: List[float], timestamps: List[float]) -> TimePattern:
        """Detect time pattern for an API call"""
        if not intervals:
            return TimePattern(api_name, rank, stream, 0, 0, 0, 0, 0, 0, 0, len(timestamps), 0, 'unknown')
        
        # Calculate statistical measures
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        
        # Calculate percentiles
        p50_interval = np.percentile(intervals, 50)
        p95_interval = np.percentile(intervals, 95)
        p99_interval = np.percentile(intervals, 99)
        
        # Determine pattern type
        pattern_type = self._classify_pattern(intervals, mean_interval, std_interval)
        
        # Calculate total duration
        total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        return TimePattern(
            api_name=api_name,
            rank=rank,
            stream=stream,
            mean_interval=mean_interval,
            std_interval=std_interval,
            min_interval=min_interval,
            max_interval=max_interval,
            p50_interval=p50_interval,
            p95_interval=p95_interval,
            p99_interval=p99_interval,
            call_count=len(timestamps),
            total_duration=total_duration,
            pattern_type=pattern_type
        )
    
    def _classify_pattern(self, intervals: List[float], mean_interval: float, std_interval: float) -> str:
        """Classify the time pattern type"""
        if len(intervals) < 3:
            return 'insufficient_data'
        
        # Calculate coefficient of variation
        cv = std_interval / mean_interval if mean_interval > 0 else float('inf')
        
        # Check for regularity
        if cv < 0.1:  # Very regular
            return 'regular'
        elif cv < 0.3:  # Moderately regular
            return 'moderately_regular'
        elif cv < 0.5:  # Somewhat irregular
            return 'irregular'
        else:  # Highly irregular
            return 'highly_irregular'
    
    def _detect_anomalies(self, api_name: str, rank: int, stream: str, 
                         intervals: List[float], timestamps: List[float], 
                         pattern: TimePattern) -> List[TimeAnomaly]:
        """Detect time anomalies"""
        anomalies = []
        
        if len(intervals) < 3:
            return anomalies
        
        # Method 1: Z-score based anomaly detection
        z_scores = self._calculate_z_scores(intervals, pattern.mean_interval, pattern.std_interval)
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > 3:  # 3-sigma rule
                anomaly = TimeAnomaly(
                    api_name=api_name,
                    rank=rank,
                    stream=stream,
                    anomaly_type='interval_outlier',
                    severity=self._determine_severity(abs(z_score)),
                    description=f"Interval {i+1} has z-score {z_score:.2f} (expected: ±3)",
                    timestamp=timestamps[i+1],
                    expected_interval=pattern.mean_interval,
                    actual_interval=intervals[i],
                    z_score=z_score
                )
                anomalies.append(anomaly)
        
        # Method 2: IQR based anomaly detection
        q1 = np.percentile(intervals, 25)
        q3 = np.percentile(intervals, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for i, interval in enumerate(intervals):
            if interval < lower_bound or interval > upper_bound:
                # Check if this anomaly wasn't already detected by z-score
                if not any(a.anomaly_type == 'interval_outlier' and 
                          a.timestamp == timestamps[i+1] for a in anomalies):
                    anomaly = TimeAnomaly(
                        api_name=api_name,
                        rank=rank,
                        stream=stream,
                        anomaly_type='interval_outlier',
                        severity=self._determine_severity(abs(interval - pattern.mean_interval) / pattern.std_interval),
                        description=f"Interval {i+1} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                        timestamp=timestamps[i+1],
                        expected_interval=pattern.mean_interval,
                        actual_interval=interval,
                        z_score=(interval - pattern.mean_interval) / pattern.std_interval
                    )
                    anomalies.append(anomaly)
        
        # Method 3: Missing calls detection
        expected_calls = self._estimate_expected_calls(pattern, timestamps[0], timestamps[-1])
        if len(timestamps) < expected_calls * 0.8:  # 20% tolerance
            anomaly = TimeAnomaly(
                api_name=api_name,
                rank=rank,
                stream=stream,
                anomaly_type='missing_call',
                severity='medium',
                description=f"Expected ~{expected_calls} calls, found {len(timestamps)}",
                timestamp=timestamps[-1],
                expected_interval=pattern.mean_interval,
                actual_interval=0,
                z_score=0
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_z_scores(self, intervals: List[float], mean: float, std: float) -> List[float]:
        """Calculate z-scores for intervals"""
        if std == 0:
            return [0] * len(intervals)
        return [(interval - mean) / std for interval in intervals]
    
    def _determine_severity(self, z_score: float) -> str:
        """Determine anomaly severity based on z-score"""
        if z_score > 5:
            return 'critical'
        elif z_score > 4:
            return 'high'
        elif z_score > 3:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_expected_calls(self, pattern: TimePattern, start_time: float, end_time: float) -> int:
        """Estimate expected number of calls based on pattern"""
        if pattern.mean_interval <= 0:
            return 1
        return int((end_time - start_time) / pattern.mean_interval) + 1
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary"""
        total_patterns = len(self.patterns)
        total_anomalies = len(self.anomalies)
        
        # Count pattern types
        pattern_types = defaultdict(int)
        for pattern in self.patterns.values():
            pattern_types[pattern.pattern_type] += 1
        
        # Count anomaly types
        anomaly_types = defaultdict(int)
        severity_counts = defaultdict(int)
        for anomaly in self.anomalies:
            anomaly_types[anomaly.anomaly_type] += 1
            severity_counts[anomaly.severity] += 1
        
        return {
            'total_patterns': total_patterns,
            'total_anomalies': total_anomalies,
            'pattern_types': dict(pattern_types),
            'anomaly_types': dict(anomaly_types),
            'severity_distribution': dict(severity_counts)
        }
    
    def print_analysis_results(self, results: Dict[str, Any]):
        """Print analysis results in a formatted way"""
        print("\n" + "="*80)
        print("TIME PATTERN ANALYSIS RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"\nSummary:")
        print(f"  • Total Patterns Analyzed: {summary['total_patterns']}")
        print(f"  • Total Anomalies Detected: {summary['total_anomalies']}")
        
        if summary['pattern_types']:
            print(f"\nPattern Types:")
            for pattern_type, count in summary['pattern_types'].items():
                print(f"  • {pattern_type}: {count}")
        
        if summary['anomaly_types']:
            print(f"\nAnomaly Types:")
            for anomaly_type, count in summary['anomaly_types'].items():
                print(f"  • {anomaly_type}: {count}")
        
        if summary['severity_distribution']:
            print(f"\nSeverity Distribution:")
            for severity, count in summary['severity_distribution'].items():
                print(f"  • {severity}: {count}")
        
        # Show top anomalies
        if results['anomalies']:
            print(f"\nTop Anomalies (by severity):")
            sorted_anomalies = sorted(results['anomalies'], 
                                    key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
                                    reverse=True)
            
            for i, anomaly in enumerate(sorted_anomalies[:10]):  # Show top 10
                print(f"  {i+1}. {anomaly.api_name} (Rank {anomaly.rank}, Stream {anomaly.stream})")
                print(f"     Type: {anomaly.anomaly_type}, Severity: {anomaly.severity}")
                print(f"     Description: {anomaly.description}")
                if anomaly.anomaly_type == 'interval_outlier':
                    print(f"     Expected: {anomaly.expected_interval:.3f}s, Actual: {anomaly.actual_interval:.3f}s")
                print()
        
        print("="*80)
    
    def generate_time_pattern_report(self, results: Dict[str, Any], output_file: str = None):
        """Generate a detailed time pattern report"""
        if output_file is None:
            output_file = "time_pattern_analysis_report.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TIME PATTERN ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Write summary
            summary = results['summary']
            f.write(f"SUMMARY\n")
            f.write(f"Total Patterns: {summary['total_patterns']}\n")
            f.write(f"Total Anomalies: {summary['total_anomalies']}\n\n")
            
            # Write detailed patterns
            f.write("DETAILED PATTERNS\n")
            f.write("-"*40 + "\n")
            for pattern_id, pattern in results['patterns'].items():
                f.write(f"API: {pattern.api_name}\n")
                f.write(f"Rank: {pattern.rank}, Stream: {pattern.stream}\n")
                f.write(f"Pattern Type: {pattern.pattern_type}\n")
                f.write(f"Call Count: {pattern.call_count}\n")
                f.write(f"Mean Interval: {pattern.mean_interval:.3f}s\n")
                f.write(f"Std Interval: {pattern.std_interval:.3f}s\n")
                f.write(f"P95 Interval: {pattern.p95_interval:.3f}s\n")
                f.write(f"Total Duration: {pattern.total_duration:.3f}s\n")
                f.write("\n")
            
            # Write anomalies
            if results['anomalies']:
                f.write("DETECTED ANOMALIES\n")
                f.write("-"*40 + "\n")
                for anomaly in results['anomalies']:
                    f.write(f"API: {anomaly.api_name}\n")
                    f.write(f"Rank: {anomaly.rank}, Stream: {anomaly.stream}\n")
                    f.write(f"Type: {anomaly.anomaly_type}\n")
                    f.write(f"Severity: {anomaly.severity}\n")
                    f.write(f"Description: {anomaly.description}\n")
                    f.write(f"Timestamp: {anomaly.timestamp}\n")
                    if anomaly.anomaly_type == 'interval_outlier':
                        f.write(f"Expected: {anomaly.expected_interval:.3f}s\n")
                        f.write(f"Actual: {anomaly.actual_interval:.3f}s\n")
                        f.write(f"Z-Score: {anomaly.z_score:.2f}\n")
                    f.write("\n")
        
        self.logger.info(f"Time pattern report saved to {output_file}")
