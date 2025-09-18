#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Hash Based Slow Node Detector
Analyzes performance differences within communication groups identified by groupHash
"""

import os
import re
import yaml
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import math

# Grubbs' test critical values for alpha=0.05 (one-sided, max outlier)
GRUBBS_CRITICAL_VALUES = {
    4: 1.463, 5: 1.672, 6: 1.822, 7: 1.938, 8: 2.032, 9: 2.110, 10: 2.176,
    11: 2.234, 12: 2.285, 13: 2.331, 14: 2.371, 15: 2.409, 16: 2.443, 17: 2.475,
    18: 2.504, 19: 2.532, 20: 2.557, 21: 2.580, 22: 2.603, 23: 2.624, 24: 2.644,
    25: 2.663, 26: 2.681, 27: 2.698, 28: 2.714, 29: 2.730, 30: 2.745
}

@dataclass
class GroupHashOp:
    """Represents a single communication operation with groupHash"""
    rank: int
    stream: str
    op_count: int
    function: str
    timestamp: float
    group_hash: str
    data_size: int

@dataclass
class GroupPerformance:
    """Performance data for a communication group identified by groupHash"""
    group_hash: str
    group_id: int  # Sequential group ID for display
    op_count: int
    ranks: List[int]
    timestamps: List[float]
    functions: List[str]
    data_sizes: List[int]
    slowest_rank: int
    slowest_time: float
    is_outlier: bool = False
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    outlier_threshold: float = 0.0
    outlier_count: int = 0

class GroupHashSlowDetector:
    """Detector for slow nodes in communication groups identified by groupHash"""
    
    def __init__(self, logs_path: str, verbose: bool = False, use_multiprocessing: bool = True):
        self.logs_path = logs_path
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.operations_by_group = defaultdict(list)  # groupHash -> List[GroupHashOp]
        self.group_performance = {}  # groupHash -> GroupPerformance
        self.rank_slow_counts = defaultdict(dict)  # rank -> group_id -> slow_count
        self.group_mapping = {}  # groupHash -> group_id (for display)
        
    def check_log_files(self) -> bool:
        """Check if log files exist and can be processed"""
        if self.verbose:
            print(f"Scanning for log files in {self.logs_path}...")
            
        log_files = glob.glob(os.path.join(self.logs_path, "*.log"))
        log_count = len(log_files)
        
        if self.verbose:
            print(f"Found {log_count} log files:")
            for i, log_file in enumerate(sorted(log_files)[:5]):
                print(f"  • {os.path.basename(log_file)}")
            if log_count > 5:
                print(f"  • ... and {log_count - 5} more files")
        
        if log_count == 0:
            print(f"Error: No log files found in {self.logs_path}")
            return False
            
        if self.verbose:
            print(f"Log file count validation passed: {log_count} files found")
            
        return True
    
    def get_rank_from_filename(self, filename: str) -> int:
        """Extract rank number from log filename"""
        patterns = [
            r'rank_(\d+)\.log',
            r'(\d+)\.log',
            r'log_(\d+)\.txt',
            r'rank(\d+)\.log'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        # If no pattern matches, try to extract any number
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])
            
        return -1
    
    def parse_log_file(self, filepath: str) -> List[GroupHashOp]:
        """Parse a single log file and extract communication operations with groupHash"""
        operations = []
        rank = self.get_rank_from_filename(os.path.basename(filepath))
        
        if rank == -1:
            if self.verbose:
                print(f"Warning: Could not extract rank from filename: {filepath}")
            return operations
        
        try:
            line_count = 0
            op_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    # Parse log line to extract operation data
                    # Format: [save_count X] [timestamp] [Rank N] Fun FunctionName Data X stream 0x... opCount N groupHash X
                    match = re.search(r'\[save_count \d+\] \[([\d.]+)\] \[Rank \d+\] Fun (\w+) Data (\d+) stream ([^\s]+) opCount (\d+)(?: groupHash (-?\d+))?', line)
                    if match:
                        timestamp = float(match.group(1))
                        function = match.group(2)
                        data_size = int(match.group(3))
                        stream = match.group(4)
                        op_count_val = int(match.group(5))
                        group_hash = match.group(6) if match.group(6) else None
                        
                        if group_hash:  # Only process operations with groupHash
                            op = GroupHashOp(
                                rank=rank,
                                stream=stream,
                                op_count=op_count_val,
                                function=function,
                                timestamp=timestamp,
                                group_hash=group_hash,
                                data_size=data_size
                            )
                            operations.append(op)
                            op_count += 1
                        
            if self.verbose:
                print(f"  Rank {rank}: Parsed {op_count} operations with groupHash from {line_count} lines")
                        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {filepath}: {e}")
                
        return operations
    
    def parse_log_file_worker(self, filepath: str) -> List[GroupHashOp]:
        """Worker function for multiprocessing log parsing"""
        return self.parse_log_file(filepath)
    
    def parse_all_logs(self) -> Dict[str, List[GroupHashOp]]:
        """Parse all log files and group operations by groupHash"""
        if not self.check_log_files():
            return {}
        
        log_files = glob.glob(os.path.join(self.logs_path, "*.log"))
        
        if self.use_multiprocessing and len(log_files) > 1:
            if self.verbose:
                print(f"Using multiprocessing to parse {len(log_files)} log files...")
            
            operations_by_rank = {}
            
            with ProcessPoolExecutor(max_workers=min(len(log_files), mp.cpu_count())) as executor:
                future_to_file = {executor.submit(self.parse_log_file_worker, filepath): filepath 
                                for filepath in log_files}
                
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        operations = future.result()
                        rank = self.get_rank_from_filename(os.path.basename(filepath))
                        if rank != -1:
                            operations_by_rank[rank] = operations
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing {filepath}: {e}")
        else:
            if self.verbose:
                print(f"Parsing {len(log_files)} log files sequentially...")
            
            operations_by_rank = {}
            for filepath in log_files:
                operations = self.parse_log_file(filepath)
                rank = self.get_rank_from_filename(os.path.basename(filepath))
                if rank != -1:
                    operations_by_rank[rank] = operations
        
        # Group operations by groupHash
        for rank, operations in operations_by_rank.items():
            for op in operations:
                self.operations_by_group[op.group_hash].append(op)
        
        if self.verbose:
            print(f"Grouped operations into {len(self.operations_by_group)} communication groups")
            for group_hash, ops in self.operations_by_group.items():
                ranks_in_group = sorted(set(op.rank for op in ops))
                print(f"  Group {group_hash}: {len(ops)} operations, ranks {ranks_in_group}")
        
        return self.operations_by_group
    
    def analyze_group_performance(self) -> Dict[str, GroupPerformance]:
        """Analyze performance for each communication group"""
        if not self.operations_by_group:
            print("No operations found. Please parse logs first.")
            return {}
        
        if self.verbose:
            print("Analyzing performance for each communication group...")
        
        # Create sequential group IDs for display
        unique_groups = sorted(self.operations_by_group.keys())
        for i, group_hash in enumerate(unique_groups):
            self.group_mapping[group_hash] = i + 1
        
        for group_hash, operations in self.operations_by_group.items():
            if len(operations) < 2:
                continue  # Need at least 2 operations to compare
            
            # Group operations by function and op_count to find same operations
            op_groups = defaultdict(list)
            for op in operations:
                # Group by function + op_count combination
                op_key = (op.function, op.op_count)
                op_groups[op_key].append((op.rank, op.timestamp))
            
            # For each operation group, find the slowest rank
            for op_key, rank_timestamps in op_groups.items():
                if len(rank_timestamps) < 2:
                    continue  # Need at least 2 ranks for comparison
                
                function_name, op_count = op_key
                ranks = [rt[0] for rt in rank_timestamps]
                timestamps = [rt[1] for rt in rank_timestamps]
                
                # Find slowest rank
                slowest_idx = np.argmax(timestamps)
                slowest_rank = ranks[slowest_idx]
                slowest_time = timestamps[slowest_idx]
                
                # Create GroupPerformance object
                group_perf = GroupPerformance(
                    group_hash=group_hash,
                    group_id=self.group_mapping[group_hash],
                    op_count=op_count,
                    ranks=ranks,
                    timestamps=timestamps,
                    functions=[function_name],
                    data_sizes=[op.data_size for op in operations if op.function == function_name and op.op_count == op_count],
                    slowest_rank=slowest_rank,
                    slowest_time=slowest_time,
                    is_outlier=True,  # Always mark as outlier since we're finding the slowest
                    q1=0.0,
                    q3=0.0,
                    iqr=0.0,
                    outlier_threshold=slowest_time,
                    outlier_count=1
                )
                
                # Store with unique key to avoid overwriting
                unique_key = f"{group_hash}_{function_name}_{op_count}"
                self.group_performance[unique_key] = group_perf
                
                # Count slow operations for each rank
                # Only the slowest rank gets counted as slow
                group_id = self.group_mapping[group_hash]
                if group_id not in self.rank_slow_counts[slowest_rank]:
                    self.rank_slow_counts[slowest_rank][group_id] = 0
                self.rank_slow_counts[slowest_rank][group_id] += 1
        
        if self.verbose:
            print(f"Performance analysis completed for {len(self.group_performance)} groups")
        
        return self.group_performance
    
    def generate_slow_rank_matrix(self) -> List[List[object]]:
        """Generate a matrix showing slow counts for each rank and group.
        Rules:
        - Only ranks that participated in a group (count > 0) are considered for that group's min normalization.
        - Non-participant ranks are set to None (NaN) for that group and excluded from min computation and totals.
        - Each group's values are normalized by subtracting the per-group minimum among participants.
        Returns rows of: [rank, v_group1, v_group2, ...] where values are int or None.
        """
        if not self.group_performance:
            print("No performance data found. Please analyze performance first.")
            return []
        
        # Get all ranks and groups
        all_ranks = set()
        for group_perf in self.group_performance.values():
            all_ranks.update(group_perf.ranks)
        
        all_ranks = sorted(all_ranks)
        all_groups = sorted(self.group_mapping.values())
        
        # Map group_id back to group_hash to determine true participants (ranks where this hash appears at least once)
        id_to_hash: Dict[int, str] = {gid: gh for gh, gid in self.group_mapping.items()}
        group_min_values: Dict[int, int] = {}
        group_participants: Dict[int, List[int]] = {}
        for group_id in all_groups:
            gh = id_to_hash.get(group_id)
            ops = self.operations_by_group.get(gh, []) if gh is not None else []
            participants_set = {op.rank for op in ops}
            participants = sorted(participants_set)
            group_participants[group_id] = participants
            if participants:
                group_min_values[group_id] = min(self.rank_slow_counts[r].get(group_id, 0) for r in participants)
            else:
                group_min_values[group_id] = 0
        
        # Create normalized matrix with None for non-participants
        matrix_data = []
        for rank in all_ranks:
            row = [rank]
            for group_id in all_groups:
                slow_count = self.rank_slow_counts[rank].get(group_id, 0)
                # participant if rank had at least one op for this groupHash
                if rank not in group_participants.get(group_id, []):
                    row.append(None)  # true non-participant
                else:
                    normalized_count = slow_count - group_min_values[group_id]
                    row.append(normalized_count)
            matrix_data.append(row)
        
        return matrix_data
    
    def print_slow_rank_summary(self):
        """Print a summary of slow rank analysis"""
        if not self.group_performance:
            print("No performance data found. Please analyze performance first.")
            return
        
        print("\n" + "="*80)
        print("GROUP HASH BASED SLOW RANK ANALYSIS")
        print("="*80)
        
        # Print group information
        print(f"\nFound {len(self.group_performance)} communication groups:")
        for group_hash, group_perf in sorted(self.group_performance.items(), key=lambda x: x[1].group_id):
            print(f"  Group {group_perf.group_id}: {len(group_perf.ranks)} ranks, {group_perf.op_count} operations")
            print(f"    Ranks: {sorted(group_perf.ranks)}")
            print(f"    Functions: {list(set(group_perf.functions))}")
            print(f"    Data sizes: {list(set(group_perf.data_sizes))}")
            print(f"    Slowest rank: {group_perf.slowest_rank} (time: {group_perf.slowest_time:.6f})")
            if group_perf.is_outlier:
                print(f"    Outliers detected: {group_perf.outlier_count} ranks above threshold")
            print()
        
        # Print slow rank matrix
        print("SLOW RANK MATRIX (Rank vs Group)")
        print("-" * 80)
        matrix_data = self.generate_slow_rank_matrix()
        if matrix_data:
            # Print header
            all_groups = sorted(self.group_mapping.values())
            header = f"{'Rank':<6}"
            for group_id in all_groups:
                header += f"{'Group_'+str(group_id):<8}"
            print(header)
            print("-" * (6 + 8 * len(all_groups)))
            
            # Print matrix rows
            for row in matrix_data:
                rank = row[0]
                slow_counts = row[1:]
                row_str = f"{rank:<6}"
                for count in slow_counts:
                    row_str += f"{count:<8}"
                print(row_str)
        
        # Print summary statistics
        print(f"\nSUMMARY:")
        print(f"  Total groups analyzed: {len(self.group_performance)}")
        print(f"  Total ranks: {len(set().union(*[gp.ranks for gp in self.group_performance.values()]))}")
        
        # Find ranks with most slow operations
        total_slow_counts = {}
        for rank, group_counts in self.rank_slow_counts.items():
            total_slow_counts[rank] = sum(group_counts.values())
        
        if total_slow_counts:
            max_slow_rank = max(total_slow_counts.items(), key=lambda x: x[1])
            print(f"  Rank with most slow operations: {max_slow_rank[0]} ({max_slow_rank[1]} total)")
        
        print("="*80)
    
    def print_parallel_style_summary(self):
        """Print summary in parallel slow node analysis style"""
        if not self.group_performance:
            print("No performance data found. Please analyze performance first.")
            return
        
        print("\n" + "="*80)
        print("GROUP HASH BASED SLOW RANK ANALYSIS (Parallel Style)")
        print("="*80)
        
        # Calculate statistics
        total_groups = len(self.group_performance)
        total_slow_picks = sum(1 for gp in self.group_performance.values() if gp.is_outlier)
        
        # Get all ranks
        all_ranks = set()
        for group_perf in self.group_performance.values():
            all_ranks.update(group_perf.ranks)
        all_ranks = sorted(all_ranks)
        
        # Calculate slow counts and rates for each rank
        rank_stats = {}
        for rank in all_ranks:
            slow_counts_by_group = {}
            total_slow_count = 0
            
            for group_hash, group_perf in self.group_performance.items():
                group_id = self.group_mapping[group_hash]
                slow_count = self.rank_slow_counts[rank].get(group_id, 0)
                slow_counts_by_group[group_id] = slow_count
                total_slow_count += slow_count
            
            # Calculate participation rate (how many groups this rank participated in)
            participations = sum(1 for gp in self.group_performance.values() if rank in gp.ranks)
            
            rank_stats[rank] = {
                'slow_counts': slow_counts_by_group,
                'total_slow': total_slow_count,
                'participations': participations
            }
        
        # Print configuration and summary
        print(f"  • Groups Analyzed: {total_groups}")
        print(f"  • Slow Picks: {total_slow_picks}")
        
        # Print slow counts and rates
        print("  • Normalized Slow Counts / Rate (relative to min per type; rate = raw/participations):")
        print(f"    {'Rank':<6} {'Total':<8}")
        print("    " + "-" * 80)
        
        for rank in all_ranks:
            stats = rank_stats[rank]
            total_slow = stats['total_slow']
            participations = stats['participations']
            rate = (total_slow / participations * 100) if participations > 0 else 0.0
            
            print(f"    {rank:<6} {total_slow:<8} ({rate:.1f}%)")
        
        # Print cumulative slow time (if available)
        print("  • Cumulative Slow Time (seconds):")
        print(f"    {'Rank':<6} {'Total':<8}")
        print("    " + "-" * 80)
        
        for rank in all_ranks:
            # Calculate total slow time for this rank
            total_slow_time = 0.0
            for group_hash, group_perf in self.group_performance.items():
                if rank in group_perf.ranks:
                    # Find the slowest time in this group
                    group_slowest_time = group_perf.slowest_time
                    # If this rank is the slowest, add the time difference
                    if rank == group_perf.slowest_rank:
                        # Calculate time difference from median
                        timestamps = group_perf.timestamps
                        if len(timestamps) > 1:
                            median_time = np.median(timestamps)
                            time_diff = group_slowest_time - median_time
                            total_slow_time += time_diff
            
            print(f"    {rank:<6} {total_slow_time:<8.6f}")
        
        print("="*80)
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Group Hash based slow rank analysis...")
        
        # Parse logs
        if not self.parse_all_logs():
            print("Failed to parse logs. Exiting.")
            return
        
        # Analyze performance
        if not self.analyze_group_performance():
            print("Failed to analyze performance. Exiting.")
            return
        
        # Print results in both formats
        self.print_slow_rank_summary()
        self.print_parallel_style_summary()
        
        return self.group_performance

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Group Hash Based Slow Rank Detector')
    parser.add_argument('logs_path', help='Path to directory containing log files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-mp', action='store_true', help='Disable multiprocessing')
    
    args = parser.parse_args()
    
    detector = GroupHashSlowDetector(
        logs_path=args.logs_path,
        verbose=args.verbose,
        use_multiprocessing=not args.no_mp
    )
    
    detector.run_analysis()

if __name__ == "__main__":
    main()
