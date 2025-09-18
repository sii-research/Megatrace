#!/usr/bin/env python3
"""
Parallel Slow Node Detector
Analyzes performance differences within TP, PP, DP communication groups
"""

import os
import re
import yaml
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import math

# Grubbs' test critical values for alpha=0.05 (one-sided, max outlier)
# Values for n=4 to n=30
GRUBBS_CRITICAL_VALUES = {
    4: 1.463, 5: 1.672, 6: 1.822, 7: 1.938, 8: 2.032, 9: 2.110, 10: 2.176,
    11: 2.234, 12: 2.285, 13: 2.331, 14: 2.371, 15: 2.409, 16: 2.443, 17: 2.475,
    18: 2.504, 19: 2.532, 20: 2.557, 21: 2.580, 22: 2.603, 23: 2.624, 24: 2.644,
    25: 2.663, 26: 2.681, 27: 2.698, 28: 2.714, 29: 2.730, 30: 2.745
}

class CommunicationOp:
    """Represents a single communication operation"""
    def __init__(self, rank: int, stream: str, op_count: int, function: str, 
                 timestamp: float, tp_group: int, pp_stage: int, dp_group: int, group_hash: str = None):
        self.rank = rank
        self.stream = stream
        self.op_count = op_count
        self.function = function
        self.timestamp = timestamp
        self.tp_group = tp_group
        self.pp_stage = pp_stage
        self.dp_group = dp_group
        self.group_hash = group_hash

@dataclass
class GroupPerformance:
    """Performance data for a communication group"""
    group_type: str  # 'TP', 'PP', 'DP'
    group_id: int
    op_count: int
    ranks: List[int]
    timestamps: List[float]
    slowest_rank: int
    slowest_time: float
    is_outlier: bool = False
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    outlier_threshold: float = 0.0
    outlier_count: int = 0

class ParallelSlowDetector:
    """Detector for slow nodes in parallel communication groups"""
    
    def __init__(self, logs_path: str, config_path: str = 'config.yaml', verbose: bool = False, use_multiprocessing: bool = True):
        self.logs_path = logs_path
        self.config_path = config_path
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.config = None
        self.tp_size = 0
        self.pp_size = 0
        self.dp_size = 0
        self.world_size = 0
        
    def load_config(self) -> bool:
        """Load configuration from config.yaml"""
        try:
            if self.verbose:
                print(f"Loading configuration from {self.config_path}...")
                
            if not os.path.exists(self.config_path):
                print(f"Error: Config file '{self.config_path}' not found!")
                return False
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.tp_size = self.config.get('TP', 2)
            self.pp_size = self.config.get('PP', 2)
            self.world_size = self.config.get('world_size', 8)
            
            # Calculate DP size
            self.dp_size = self.world_size // (self.tp_size * self.pp_size)
            
            if self.dp_size == 0:
                print(f"Error: Invalid configuration - DP_SIZE cannot be 0")
                return False
                
            if self.verbose:
                print(f"Configuration loaded successfully:")
                print(f"  • TP_SIZE: {self.tp_size}")
                print(f"  • PP_SIZE: {self.pp_size}")
                print(f"  • DP_SIZE: {self.dp_size} (calculated)")
                print(f"  • World Size: {self.world_size}")
                print(f"  • Multiprocessing: {'Enabled' if self.use_multiprocessing else 'Disabled'}")
                
            return True
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def check_log_files(self) -> bool:
        """Check if log files match world_size configuration"""
        if self.verbose:
            print(f"Scanning for log files in {self.logs_path}...")
            
        log_files = glob.glob(os.path.join(self.logs_path, "*.log"))
        log_count = len(log_files)
        
        if self.verbose:
            print(f"Found {log_count} log files:")
            for i, log_file in enumerate(sorted(log_files)[:5]):  # Show first 5 files
                print(f"  • {os.path.basename(log_file)}")
            if log_count > 5:
                print(f"  • ... and {log_count - 5} more files")
        
        if log_count != self.world_size:
            print(f"Error: Log file count ({log_count}) doesn't match world_size ({self.world_size})")
            return False
            
        if self.verbose:
            print(f"Log file count validation passed: {log_count} files match world_size configuration")
            
        return True
    
    def get_rank_from_filename(self, filename: str) -> int:
        """Extract rank number from log filename"""
        # Common patterns: rank_0.log, 0.log, log_0.txt, etc.
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
    
    def calculate_parallel_positions(self, rank: int) -> Tuple[int, int, int, int, int, int]:
        """Calculate TP, PP, DP positions for a given rank"""
        tp_pos = rank % self.tp_size
        tp_group = rank // self.tp_size
        
        pp_stage = rank // (self.world_size // self.pp_size)
        pp_group = rank % (self.world_size // self.pp_size)
        
        dp_pos = (rank // self.tp_size) % self.dp_size
        dp_group = (rank // (self.tp_size * self.dp_size)) * self.tp_size + rank % self.tp_size
        
        return tp_pos, tp_group, pp_stage, pp_group, dp_pos, dp_group
    
    def parse_log_file(self, filepath: str) -> List[CommunicationOp]:
        """Parse a single log file and extract communication operations"""
        operations = []
        rank = self.get_rank_from_filename(os.path.basename(filepath))
        
        if rank == -1:
            if self.verbose:
                print(f"Warning: Could not extract rank from filename: {filepath}")
            return operations
        
        tp_pos, tp_group, pp_stage, pp_group, dp_pos, dp_group = self.calculate_parallel_positions(rank)
        
        try:
            line_count = 0
            op_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    # Parse log line to extract operation data
                    # Format: [save_count X] [timestamp] [Rank N] Fun FunctionName Data X stream 0x... opCount N groupHash X
                    match = re.search(r'\[save_count \d+\] \[([\d.]+)\] \[Rank \d+\] Fun (\w+) Data \d+ stream ([^\s]+) opCount (\d+)(?: groupHash (-?\d+))?', line)
                    if match:
                        timestamp = float(match.group(1))
                        function = match.group(2)
                        stream = match.group(3)
                        op_count_val = int(match.group(4))
                        group_hash = match.group(5) if match.group(5) else None
                        
                        op = CommunicationOp(
                            rank=rank,
                            stream=stream,
                            op_count=op_count_val,
                            function=function,
                            timestamp=timestamp,
                            tp_group=tp_group,
                            pp_stage=pp_stage,
                            dp_group=dp_group,
                            group_hash=group_hash
                        )
                        operations.append(op)
                        op_count += 1
                        
            if self.verbose:
                print(f"  Rank {rank}: Parsed {op_count} operations from {line_count} lines")
                        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {filepath}: {e}")
                
        return operations
    
    def parse_log_file_worker(self, filepath: str) -> List[CommunicationOp]:
        """Worker function for multiprocessing log parsing"""
        return self.parse_log_file(filepath)
    
    def parse_all_logs(self) -> List[CommunicationOp]:
        """Parse all log files and extract communication operations"""
        log_files = glob.glob(os.path.join(self.logs_path, "*.log"))
        
        if self.verbose:
            print(f"Starting to parse {len(log_files)} log files...")
            if self.use_multiprocessing:
                cpu_count = mp.cpu_count()
                print(f"Using multiprocessing with {cpu_count} CPU cores")
            else:
                print("Using single-threaded processing")
        
        all_operations = []
        start_time = time.time()
        
        if self.use_multiprocessing and len(log_files) > 1:
            # Use multiprocessing for parallel parsing
            with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(log_files))) as executor:
                # Submit all parsing tasks
                future_to_file = {executor.submit(self.parse_log_file_worker, log_file): log_file 
                                for log_file in log_files}
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_file):
                    filepath = future_to_file[future]
                    try:
                        operations = future.result()
                        all_operations.extend(operations)
                        completed += 1
                        
                        if self.verbose:
                            progress = (completed / len(log_files)) * 100
                            print(f"Progress: {completed}/{len(log_files)} files ({progress:.1f}%) - {os.path.basename(filepath)}")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing {filepath}: {e}")
        else:
            # Single-threaded processing
            for i, log_file in enumerate(log_files):
                if self.verbose:
                    progress = ((i + 1) / len(log_files)) * 100
                    print(f"Progress: {i + 1}/{len(log_files)} files ({progress:.1f}%) - {os.path.basename(log_file)}")
                    
                operations = self.parse_log_file(log_file)
                all_operations.extend(operations)
        
        end_time = time.time()
        
        if self.verbose:
            print(f"Parsing completed in {end_time - start_time:.2f} seconds")
            print(f"Total: {len(all_operations)} communication operations from {len(log_files)} log files")
            
        return all_operations
    
    def group_by_communication_type(self, operations: List[CommunicationOp]) -> Dict[str, Dict]:
        """Group operations by communication type (TP, PP, DP)"""
        if self.verbose:
            print(f"Grouping {len(operations)} operations by communication type...")
        
        groups = {
            'TP': defaultdict(list),
            'PP': defaultdict(list),
            'DP': defaultdict(list)
        }
        
        for op in operations:
            # Group by TP: same tp_group, same op_count, same stream
            tp_key = (op.tp_group, op.op_count, op.stream)
            groups['TP'][tp_key].append(op)
            
            # Group by PP: same pp_stage, same op_count, same stream
            pp_key = (op.pp_stage, op.op_count, op.stream)
            groups['PP'][pp_key].append(op)
            
            # Group by DP: same dp_group, same op_count, same stream
            dp_key = (op.dp_group, op.op_count, op.stream)
            groups['DP'][dp_key].append(op)
        
        if self.verbose:
            print(f"Grouping completed:")
            print(f"  • TP Groups: {len(groups['TP'])}")
            print(f"  • PP Groups: {len(groups['PP'])}")
            print(f"  • DP Groups: {len(groups['DP'])}")
            
            # Show some example groups
            for comm_type in ['TP', 'PP', 'DP']:
                if groups[comm_type]:
                    example_groups = list(groups[comm_type].items())[:3]
                    print(f"  • {comm_type} Example Groups:")
                    for key, ops in example_groups:
                        ranks = sorted(set(op.rank for op in ops))
                        print(f"    - Group {key}: {len(ops)} ops, Ranks {ranks}")
        
        return groups
    
    def analyze_group_performance(self, group_ops: List[CommunicationOp], group_type: str, group_id: int) -> GroupPerformance:
        """Analyze performance within a communication group.
        - Use Grubbs' test for groups with 4-29 members (if SciPy available)
        - Otherwise fall back to IQR method
        """
        if not group_ops:
            return None
            
        # Extract timestamps and ranks
        timestamps = [op.timestamp for op in group_ops]
        ranks = [op.rank for op in group_ops]
        
        # Calculate quartiles and IQR
        sorted_timestamps = sorted(timestamps)
        n = len(sorted_timestamps)
        
        if n < 4:
            # If less than 4 operations, use simple max method
            slowest_op = max(group_ops, key=lambda x: x.timestamp)
            return GroupPerformance(
                group_type=group_type,
                group_id=group_id,
                op_count=group_ops[0].op_count,
                ranks=ranks,
                timestamps=timestamps,
                slowest_rank=slowest_op.rank,
                slowest_time=slowest_op.timestamp
            )
        
        # Calculate quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_timestamps[q1_idx]
        q3 = sorted_timestamps[q3_idx]
        iqr = q3 - q1

        # Prefer Grubbs' test when group size is between 4 and 29
        if 4 <= n <= 30:
            mean_val = float(np.mean(timestamps))
            std_val = float(np.std(timestamps, ddof=1))
            if std_val > 0:
                # Get critical value from lookup table
                g_crit = GRUBBS_CRITICAL_VALUES.get(n, 2.0)  # Default to 2.0 if not in table
                max_idx = int(np.argmax(timestamps))
                g_stat = (timestamps[max_idx] - mean_val) / std_val
                if g_stat > g_crit:
                    return GroupPerformance(
                        group_type=group_type,
                        group_id=group_id,
                        op_count=group_ops[0].op_count,
                        ranks=ranks,
                        timestamps=timestamps,
                        slowest_rank=ranks[max_idx],
                        slowest_time=timestamps[max_idx],
                        is_outlier=True,
                        q1=q1,
                        q3=q3,
                        iqr=iqr,
                        outlier_threshold=mean_val + g_crit * std_val,
                        outlier_count=1
                    )
        
        # Define outlier threshold (use 1.5 * IQR as standard statistical method)
        outlier_threshold = q3 + 1.5 * iqr
        
        # Find outliers (slow operations)
        outliers = []
        for i, timestamp in enumerate(timestamps):
            if timestamp > outlier_threshold:
                outliers.append((timestamp, ranks[i]))
        
        # Debug: Show statistics for first few groups (only if outliers found)
        if self.verbose and group_id < 3 and outliers:
            print(f"    Debug Group {group_type} {group_id}: Found {len(outliers)} outliers")
            print(f"      Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}, threshold={outlier_threshold:.6f}")
            print(f"      Min={min(timestamps):.6f}, Max={max(timestamps):.6f}, Range={max(timestamps)-min(timestamps):.6f}")
        
        if outliers:
            # Return the slowest outlier
            slowest_outlier = max(outliers, key=lambda x: x[0])
            return GroupPerformance(
                group_type=group_type,
                group_id=group_id,
                op_count=group_ops[0].op_count,
                ranks=ranks,
                timestamps=timestamps,
                slowest_rank=slowest_outlier[1],
                slowest_time=slowest_outlier[0],
                is_outlier=True,
                q1=q1,
                q3=q3,
                iqr=iqr,
                outlier_threshold=outlier_threshold,
                outlier_count=len(outliers)
            )
        else:
            # No outliers found, return None to indicate no slow operations
            return None
    
    def calculate_slow_scores(self, groups: Dict[str, Dict]) -> Dict[int, Dict[str, int]]:
        """Calculate slow scores for each rank in each communication type using IQR method"""
        if self.verbose:
            print(f"Calculating slow scores for communication groups using IQR method...")
        
        # Initialize scores: {rank: {'TP': score, 'PP': score, 'DP': score}}
        scores = defaultdict(lambda: {'TP': 0, 'PP': 0, 'DP': 0})
        
        total_groups_analyzed = 0
        total_outliers_found = 0
        
        for comm_type, type_groups in groups.items():
            if self.verbose:
                print(f"  Analyzing {comm_type} groups...")
            
            groups_analyzed = 0
            outliers_found = 0
            
            for group_key, group_ops in type_groups.items():
                if len(group_ops) > 1:  # Only analyze groups with multiple ranks
                    performance = self.analyze_group_performance(group_ops, comm_type, group_key[0])
                    if performance:
                        groups_analyzed += 1
                        
                        if performance.is_outlier:
                            # Only count outliers as slow operations
                            scores[performance.slowest_rank][comm_type] += 1
                            outliers_found += 1
                            
                            if self.verbose and outliers_found <= 5:  # Show first 5 outliers
                                print(f"    {comm_type} Group {group_key}: Outlier rank {performance.slowest_rank} "
                                      f"(op_count {performance.op_count}, time {performance.slowest_time:.3f})")
                                print(f"      Q1={performance.q1:.3f}, Q3={performance.q3:.3f}, IQR={performance.iqr:.3f}, "
                                      f"threshold={performance.outlier_threshold:.3f}, outliers={performance.outlier_count}")
            
            total_groups_analyzed += groups_analyzed
            total_outliers_found += outliers_found
            
            if self.verbose:
                print(f"  {comm_type}: Analyzed {groups_analyzed} groups, found {outliers_found} outliers")
        
        if self.verbose:
            print(f"Score calculation completed: {total_groups_analyzed} groups analyzed, {total_outliers_found} outliers found")
        
        return scores
    
    def normalize_scores(self, scores: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        """Normalize scores by subtracting minimum value"""
        normalized = {}
        
        for comm_type in ['TP', 'PP', 'DP']:
            type_scores = [scores[rank][comm_type] for rank in scores]
            if type_scores:
                min_score = min(type_scores)
                for rank in scores:
                    if rank not in normalized:
                        normalized[rank] = {'TP': 0, 'PP': 0, 'DP': 0}
                    normalized[rank][comm_type] = scores[rank][comm_type] - min_score
        
        return normalized
    
    def analyze_parallel_slow_nodes(self) -> Dict:
        """Main analysis function"""
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting parallel slow node analysis...")
            print(f"{'='*60}")
        
        # Load configuration
        if not self.load_config():
            return None
            
        # Check log files
        if not self.check_log_files():
            return None
            
        # Parse all log files
        operations = self.parse_all_logs()
        if not operations:
            print("No communication operations found in log files")
            return None
            
        # Group by communication type
        groups = self.group_by_communication_type(operations)
        
        # Calculate slow scores
        scores = self.calculate_slow_scores(groups)
        
        # Normalize scores
        if self.verbose:
            print(f"Normalizing scores...")
        normalized_scores = self.normalize_scores(scores)
        
        end_time = time.time()
        
        if self.verbose:
            print(f"Analysis completed in {end_time - start_time:.2f} seconds")
            print(f"{'='*60}")
        
        # Prepare results
        results = {
            'configuration': {
                'tp_size': self.tp_size,
                'pp_size': self.pp_size,
                'dp_size': self.dp_size,
                'world_size': self.world_size
            },
            'analysis_summary': {
                'total_operations': len(operations),
                'tp_groups': len(groups['TP']),
                'pp_groups': len(groups['PP']),
                'dp_groups': len(groups['DP'])
            },
            'raw_scores': scores,
            'normalized_scores': normalized_scores,
            'groups': groups
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print analysis results"""
        if not results:
            return
            
        config = results['configuration']
        summary = results['analysis_summary']
        normalized_scores = results['normalized_scores']
        
        print(f"\n{'='*80}")
        print("PARALLEL SLOW NODE ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        print(f"\nConfiguration:")
        print(f"  • TP_SIZE: {config['tp_size']}")
        print(f"  • PP_SIZE: {config['pp_size']}")
        print(f"  • DP_SIZE: {config['dp_size']}")
        print(f"  • World Size: {config['world_size']}")
        
        print(f"\nAnalysis Summary:")
        print(f"  • Total Operations: {summary['total_operations']}")
        print(f"  • TP Groups: {summary['tp_groups']}")
        print(f"  • PP Groups: {summary['pp_groups']}")
        print(f"  • DP Groups: {summary['dp_groups']}")
        print(f"  • Analysis Method: IQR (Interquartile Range) + Grubbs' test (for 4-30 sized groups)")
        print(f"  • IQR Threshold: Q3 + 1.5 * IQR (standard)")
        print(f"  • Grubbs: one-sided max outlier at alpha=0.05 (lookup table)")
        
        print(f"\nNormalized Slow Scores (relative to minimum):")
        print(f"{'Rank':<6} {'TP':<8} {'PP':<8} {'DP':<8} {'Total':<8}")
        print("-" * 40)
        
        for rank in sorted(normalized_scores.keys()):
            scores = normalized_scores[rank]
            total = sum(scores.values())
            print(f"{rank:<6} {scores['TP']:<8} {scores['PP']:<8} {scores['DP']:<8} {total:<8}")
        
        # Find ranks with highest scores
        rank_totals = {rank: sum(scores.values()) for rank, scores in normalized_scores.items()}
        if rank_totals:
            max_total = max(rank_totals.values())
            slowest_ranks = [rank for rank, total in rank_totals.items() if total == max_total]
            
            print(f"\nSlowest Ranks (highest relative scores):")
            for rank in slowest_ranks:
                scores = normalized_scores[rank]
                print(f"  • Rank {rank}: TP={scores['TP']}, PP={scores['PP']}, DP={scores['DP']} (Total={rank_totals[rank]})")
        
        print(f"\n{'='*80}")

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel Slow Node Detector')
    parser.add_argument('--logs-path', required=True, help='Path to logs directory')
    parser.add_argument('--config-path', default='config.yaml', help='Path to config file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing (use single thread)')
    
    args = parser.parse_args()
    
    detector = ParallelSlowDetector(
        args.logs_path, 
        args.config_path, 
        args.verbose,
        use_multiprocessing=not args.no_multiprocessing
    )
    results = detector.analyze_parallel_slow_nodes()
    detector.print_results(results)

if __name__ == "__main__":
    main()
