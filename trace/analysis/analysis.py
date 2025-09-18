#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Training Log Analyzer - Analyze all log files in specified path
Support hang analysis and slow analysis functionality
"""

import os
import sys
import argparse
import logging
import re
import gzip
import bz2
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import defaultdict, namedtuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import math

# Import time pattern analyzer
try:
    from time_pattern_analyzer import TimePatternAnalyzer
except ImportError:
    TimePatternAnalyzer = None


# Define log entry data structure
class LogEntry:
    """Single log entry data structure"""
    def __init__(self, raw_line: str, save_count: int, timestamp: float, 
                 rank: int, function: str, data_size: int, stream: str, op_count: int, group_hash: str = None):
        self.raw_line = raw_line
        self.save_count = save_count
        self.timestamp = timestamp
        self.rank = rank
        self.function = function
        self.data_size = data_size
        self.stream = stream
        self.op_count = op_count
        self.group_hash = group_hash
    
    def __str__(self):
        if self.group_hash:
            return f"[save_count {self.save_count}] [{self.timestamp}] [Rank {self.rank}] Fun {self.function} Data {self.data_size} stream {self.stream} opCount {self.op_count} groupHash {self.group_hash}"
        else:
            return f"[save_count {self.save_count}] [{self.timestamp}] [Rank {self.rank}] Fun {self.function} Data {self.data_size} stream {self.stream} opCount {self.op_count}"


class DistributedLogAnalyzer:
    """Distributed Training Log Analyzer Main Class"""
    
    def __init__(self, log_path: str, verbose: bool = False):
        """
        Initialize Distributed Log Analyzer
        
        Args:
            log_path: Path to log file or directory to analyze
            verbose: Whether to show detailed log output
        """
        self.log_path = Path(log_path)
        self.log_files = []
        self.analysis_results = {}
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Define regex pattern for log parsing
        self.log_pattern = re.compile(
            r'\[save_count (\d+)\] \[([\d.]+)\] \[Rank (\d+)\] Fun (\w+) Data (\d+) stream (0x[0-9a-fA-F]+) opCount (\d+)(?: groupHash (-?\d+))?'
        )
        
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.verbose:
            # Verbose mode: show all logs
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('hang_analysis.log', encoding='utf-8')
                ]
            )
        else:
            # Silent mode: only log to file; console shows only ERROR
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('hang_analysis.log', encoding='utf-8')
                ]
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.ERROR)
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler is not console_handler:
                    root_logger.removeHandler(handler)
            root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
        
    def discover_log_files(self) -> List[Path]:
        """
        1. Traverse log files in folder
        
        Returns:
            List containing all log file paths
        """
        log_extensions = {'.log', '.txt', '.out', '.err'}
        
        if self.log_path.is_file():
            # If it's a single file
            if self.log_path.suffix.lower() in log_extensions:
                self.log_files = [self.log_path]
                self.logger.info(f"Found single log file: {self.log_path}")
            else:
                self.logger.warning(f"File {self.log_path} is not a log file")
                return []
        elif self.log_path.is_dir():
            # If it's a directory, recursively find all log files
            self.log_files = []
            for file_path in self.log_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in log_extensions:
                    self.log_files.append(file_path)
            
            self.logger.info(f"Found {len(self.log_files)} log files in directory {self.log_path}")
        else:
            self.logger.error(f"Path {self.log_path} does not exist")
            return []
            
        return self.log_files
    
    def read_log_file(self, file_path: Path) -> List[str]:
        """Read all lines from log file"""
        lines = []
        try:
            if file_path.suffix.lower() == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            elif file_path.suffix.lower() == '.bz2':
                with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            
            # Remove newline characters from each line
            lines = [line.rstrip('\n\r') for line in lines]
            self.logger.info(f"File {file_path} read completed, total {len(lines)} lines")
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        return lines
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse single log line"""
        match = self.log_pattern.match(line)
        if match:
            save_count = int(match.group(1))
            timestamp = float(match.group(2))
            rank = int(match.group(3))
            function = match.group(4)
            data_size = int(match.group(5))
            stream = match.group(6)
            op_count = int(match.group(7))
            group_hash = match.group(8) # Get groupHash if it exists
            
            return LogEntry(line, save_count, timestamp, rank, function, data_size, stream, op_count, group_hash)
        
        return None
    
    def parse_log_files(self) -> Dict[int, List[LogEntry]]:
        """
        2. Read logs for each rank and mark rank numbers
        3. Analyze last save_count group log entries
        
        Returns:
            Log entries organized by rank dictionary
        """
        rank_entries = defaultdict(list)
        
        for log_file in self.log_files:
            self.logger.info(f"Starting to parse file: {log_file}")
            lines = self.read_log_file(log_file)
            
            for line in lines:
                entry = self.parse_log_line(line)
                if entry:
                    rank_entries[entry.rank].append(entry)
        
        # Sort entries by time for each rank
        for rank in rank_entries:
            rank_entries[rank].sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Parsing completed, found logs for {len(rank_entries)} ranks")
        for rank, entries in rank_entries.items():
            self.logger.info(f"Rank {rank}: {len(entries)} records")
        
        return rank_entries
    
    def get_last_save_count_group(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[int, List[LogEntry]]:
        """
        Get last save_count group log entries for each rank
        
        Returns:
            Last save_count group log entries for each rank
        """
        last_save_count_entries = {}
        
        for rank, entries in rank_entries.items():
            if not entries:
                continue
            
            # Find maximum save_count
            max_save_count = max(entry.save_count for entry in entries)
            
            # Filter entries from last save_count group
            last_group = [entry for entry in entries if entry.save_count == max_save_count]
            
            if last_group:
                last_save_count_entries[rank] = last_group
                self.logger.info(f"Rank {rank} last save_count {max_save_count}: {len(last_group)} records")
        
        return last_save_count_entries
    
    def group_by_stream(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[int, Dict[str, List[LogEntry]]]:
        """
        4. Group logs by stream, each stream as one group
        
        Returns:
            Log entries organized by rank and stream
        """
        stream_groups = {}
        
        for rank, entries in rank_entries.items():
            stream_groups[rank] = defaultdict(list)
            
            for entry in entries:
                stream_groups[rank][entry.stream].append(entry)
            
            # Sort entries by time within each stream
            for stream in stream_groups[rank]:
                stream_groups[rank][stream].sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Rank {rank} found {len(stream_groups[rank])} streams")
            for stream, stream_entries in stream_groups[rank].items():
                self.logger.info(f"  Stream {stream}: {len(stream_entries)} records")
        
        return stream_groups

    def classify_streams_parallel(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[int, Dict[str, set]]:
        """
        Classify streams for each rank into PP/TP/DP/Barrier according to rules:
        1) Streams with all Data < 1024 -> Barrier
        2) Streams containing only send/recv ops -> PP
        3) Streams containing only reduce or allgather/reduce; among them, the one with maximum opCount cardinality -> TP
        4) Same set as (3); the one with minimum opCount cardinality -> DP
        Note: Multiple PP streams can exist; TP/DP choose at most one each (if available). If only one reduce-class stream exists, it may be both TP and DP.
        Returns mapping: { rank: { 'PP': set(streams), 'TP': set(streams<=1), 'DP': set(streams<=1), 'BARRIER': set(streams) } }
        """
        result: Dict[int, Dict[str, set]] = {}
        for rank, entries in rank_entries.items():
            streams: Dict[str, List[LogEntry]] = defaultdict(list)
            for e in entries:
                streams[e.stream].append(e)
            barrier_set: set = set()
            pp_set: set = set()
            reduce_candidates: List[Tuple[str, int]] = []  # (stream, unique_opcount)
            for stream, s_entries in streams.items():
                if not s_entries:
                    continue
                sizes = [se.data_size for se in s_entries]
                funcs = set(se.function.lower() for se in s_entries)
                # 1) Barrier: all data < 1024
                if all(sz < 1024 for sz in sizes):
                    barrier_set.add(stream)
                # 2) PP: only send/recv
                if all(('send' in f or 'recv' in f) for f in funcs):
                    pp_set.add(stream)
                # Reduce/allgather-only candidate
                if all(('reduce' in f) or ('allgather' in f) for f in funcs):
                    unique_opcounts = len(set(se.op_count for se in s_entries))
                    reduce_candidates.append((stream, unique_opcounts))
            tp_set: set = set()
            dp_set: set = set()
            if reduce_candidates:
                # Choose by opCount cardinality
                max_stream, _ = max(reduce_candidates, key=lambda t: t[1])
                min_stream, _ = min(reduce_candidates, key=lambda t: t[1])
                tp_set.add(max_stream)
                dp_set.add(min_stream)
            result[rank] = {'PP': pp_set, 'TP': tp_set, 'DP': dp_set, 'BARRIER': barrier_set}
        # Log brief summary
        try:
            for rank in sorted(result.keys()):
                info = result[rank]
                self.logger.info(
                    f"[StreamClassify] rank={rank} PP={len(info['PP'])} TP={list(info['TP'])} DP={list(info['DP'])} BARRIER={len(info['BARRIER'])}"
                )
        except Exception:
            pass
        return result
    
    def find_last_operation_in_streams(self, stream_groups: Dict[int, Dict[str, List[LogEntry]]]) -> Dict[int, Dict[str, LogEntry]]:
        """
        5. Identify last operation in each stream group
        
        Returns:
            Last operation for each rank and stream
        """
        last_operations = {}
        
        for rank, streams in stream_groups.items():
            last_operations[rank] = {}
            
            for stream, entries in streams.items():
                if entries:
                    # Get last record
                    last_op = entries[-1]
                    last_operations[rank][stream] = last_op
                    
                    self.logger.info(f"Rank {rank} Stream {stream} last operation: {last_op.function} (opCount {last_op.op_count})")
        
        return last_operations
    
    def detect_hangs(self, last_operations: Dict[int, Dict[str, LogEntry]], 
                    stream_groups: Dict[int, Dict[str, List[LogEntry]]]) -> List[Dict[str, Any]]:
        """
        Detect hang situations based on last operations
        
        Returns:
            List of hang detection results
        """
        hangs = []
        current_time = time.time()
        
        for rank, streams in last_operations.items():
            for stream, last_op in streams.items():
                # Calculate time since last operation
                time_since_last_op = current_time - last_op.timestamp
                
                # Detect hang situations
                if time_since_last_op > 300:  # No new operation for 5 minutes considered as hang
                    hang_info = {
                        'type': 'stream_hang',
                        'rank': rank,
                        'stream': stream,
                        'last_operation': last_op.function,
                        'last_op_count': last_op.op_count,
                        'last_timestamp': last_op.timestamp,
                        'time_since_last_op': time_since_last_op,
                        'severity': 'high' if time_since_last_op > 600 else 'medium',
                        'description': f"Rank {rank} Stream {stream} no new operation for {time_since_last_op:.1f} seconds"
                    }
                    hangs.append(hang_info)
                
                # Detect specific operation hangs
                if last_op.function in ['AllReduce', 'AllGather', 'Broadcast']:
                    # Check if there are subsequent operations
                    stream_entries = stream_groups[rank][stream]
                    if len(stream_entries) >= 2:
                        last_two_ops = stream_entries[-2:]
                        interval = last_two_ops[1].timestamp - last_two_ops[0].timestamp
                        
                        if interval > 120:  # Collective operation interval over 2 minutes considered as hang
                            hang_info = {
                                'type': 'collective_operation_hang',
                                'rank': rank,
                                'stream': stream,
                                'operation': last_op.function,
                                'interval': interval,
                                'severity': 'high',
                                'description': f"Rank {rank} Stream {stream} {last_op.function} operation interval {interval:.1f} seconds"
                            }
                            hangs.append(hang_info)
        
        return hangs

    # ===== Slow analysis (simple per-rank) =====
    def analyze_slow_ranks_simple(self, rank_entries: Dict[int, List[LogEntry]]) -> Dict[str, Any]:
        """Simple per-rank slow analysis using mean+std score; returns slow ranks above P95."""
        scores_by_rank: Dict[int, float] = {}
        for rank, entries in rank_entries.items():
            if not entries:
                continue
            ts = [e.timestamp for e in entries]
            mean_t = float(np.mean(ts))
            std_t = float(np.std(ts))
            scores_by_rank[rank] = mean_t + std_t
        if not scores_by_rank:
            return {"rank_metrics": {}, "slow_ranks": [], "total_ranks": 0}
        score_values = list(scores_by_rank.values())
        threshold = float(np.percentile(score_values, 95))
        slow_ranks = [r for r, s in scores_by_rank.items() if s > threshold]
        rank_metrics = {r: {"performance_score": scores_by_rank[r]} for r in scores_by_rank}
        return {"rank_metrics": rank_metrics, "slow_ranks": slow_ranks, "total_ranks": len(scores_by_rank)}

    # ===== Parallel group slow analysis (IQR + Grubbs) =====
    @dataclass
    class CommunicationOp:
        rank: int
        stream: str
        op_count: int
        function: str
        timestamp: float
        tp_group: int
        pp_stage: int
        dp_group: int
        group_hash: str | None = None

    @dataclass
    class GroupPerformance:
        group_type: str
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

    GRUBBS_CRITICAL_VALUES = {
        4: 1.463, 5: 1.672, 6: 1.822, 7: 1.938, 8: 2.032, 9: 2.110, 10: 2.176,
        11: 2.234, 12: 2.285, 13: 2.331, 14: 2.371, 15: 2.409, 16: 2.443, 17: 2.475,
        18: 2.504, 19: 2.532, 20: 2.557, 21: 2.580, 22: 2.603, 23: 2.624, 24: 2.644,
        25: 2.663, 26: 2.681, 27: 2.698, 28: 2.714, 29: 2.730, 30: 2.745
    }

    def _calc_parallel_positions(self, rank: int, tp_size: int, pp_size: int, dp_size: int, world_size: int) -> Tuple[int,int,int,int,int,int]:
        tp_pos = rank % tp_size
        tp_group = rank // tp_size
        pp_stage = rank // (world_size // pp_size)
        pp_group = rank % (world_size // pp_size)
        dp_pos = (rank // tp_size) % dp_size
        dp_group = (rank // (tp_size * dp_size)) * tp_size + rank % tp_size
        return tp_pos, tp_group, pp_stage, pp_group, dp_pos, dp_group

    def analyze_group_performance_parallel(self, group_ops: List['DistributedLogAnalyzer.CommunicationOp'], group_type: str, group_id: int) -> Optional['DistributedLogAnalyzer.GroupPerformance']:
        if not group_ops:
            return None
        timestamps = [op.timestamp for op in group_ops]
        ranks = [op.rank for op in group_ops]
        sorted_ts = sorted(timestamps)
        n = len(sorted_ts)
        try:
            self.logger.info(
                f"[ParallelSlow][{group_type}] group_id={group_id} opCount={group_ops[0].op_count} "
                f"size={n} ranks={sorted(set(ranks))}"
            )
            # Print per-rank timestamps (after dedupe) for visibility
            per_rank = {}
            for op in group_ops:
                if (op.rank not in per_rank) or (op.timestamp > per_rank[op.rank]):
                    per_rank[op.rank] = op.timestamp
            stamp_str = ', '.join([f"{rk}:{per_rank[rk]:.6f}" for rk in sorted(per_rank.keys())])
            self.logger.info(
                f"[ParallelSlow][{group_type}] per-rank ts: {stamp_str}"
            )
        except Exception:
            pass
        if n < 2:
            return None
        if n < 4:
            slowest = max(group_ops, key=lambda x: x.timestamp)
            try:
                self.logger.info(
                    f"[ParallelSlow][{group_type}] size={n}<4, skip outlier tests; "
                    f"slowest rank={slowest.rank} ts={slowest.timestamp:.6f}"
                )
            except Exception:
                pass
            return self.GroupPerformance(group_type, group_id, group_ops[0].op_count, ranks, timestamps, slowest.rank, slowest.timestamp)
        q1 = sorted_ts[n//4]
        q3 = sorted_ts[(3*n)//4]
        iqr = q3 - q1
        try:
            self.logger.info(
                f"[ParallelSlow][{group_type}] q1={q1:.6f} q3={q3:.6f} iqr={iqr:.6f}"
            )
        except Exception:
            pass
        # Grubbs for 4..30
        if 4 <= n <= 30:
            mean_val = float(np.mean(timestamps))
            std_val = float(np.std(timestamps, ddof=1))
            if std_val > 0:
                g_crit = self.GRUBBS_CRITICAL_VALUES.get(n, 2.0)
                max_idx = int(np.argmax(timestamps))
                g_stat = (timestamps[max_idx] - mean_val) / std_val
                try:
                    self.logger.info(
                        f"[ParallelSlow][{group_type}] Grubbs n={n} mean={mean_val:.6f} std={std_val:.6f} "
                        f"g_stat={g_stat:.6f} g_crit={g_crit:.6f} max_rank={ranks[max_idx]} max_ts={timestamps[max_idx]:.6f}"
                    )
                except Exception:
                    pass
                if g_stat > g_crit:
                    try:
                        self.logger.info(
                            f"[ParallelSlow][{group_type}] Grubbs OUTLIER rank={ranks[max_idx]} "
                            f"opCount={group_ops[0].op_count}"
                        )
                    except Exception:
                        pass
                    return self.GroupPerformance(group_type, group_id, group_ops[0].op_count, ranks, timestamps, ranks[max_idx], timestamps[max_idx], True, q1, q3, iqr, mean_val + g_crit*std_val, 1)
        # IQR fallback
        threshold = q3 + 1.5 * iqr
        outliers = [(timestamps[i], ranks[i]) for i in range(n) if timestamps[i] > threshold]
        try:
            self.logger.info(
                f"[ParallelSlow][{group_type}] IQR threshold={threshold:.6f} outliers={len(outliers)}"
            )
        except Exception:
            pass
        if outliers:
            slowest = max(outliers, key=lambda x: x[0])
            try:
                self.logger.info(
                    f"[ParallelSlow][{group_type}] IQR OUTLIER slowest_rank={slowest[1]} ts={slowest[0]:.6f}"
                )
            except Exception:
                pass
            return self.GroupPerformance(group_type, group_id, group_ops[0].op_count, ranks, timestamps, slowest[1], slowest[0], True, q1, q3, iqr, threshold, len(outliers))
        return None

    def analyze_parallel_slow_nodes(self, rank_entries: Dict[int, List[LogEntry]], tp_size: int, pp_size: int, world_size: int) -> Dict[str, Any]:
        dp_size = world_size // (tp_size * pp_size)
        if dp_size == 0:
            return {"error": "Invalid configuration - DP_SIZE cannot be 0"}
        # Build CommunicationOp list
        comm_ops: List[DistributedLogAnalyzer.CommunicationOp] = []
        for rank, entries in rank_entries.items():
            for e in entries:
                _, tp_group, pp_stage, pp_group, _, dp_group = self._calc_parallel_positions(rank, tp_size, pp_size, dp_size, world_size)
                comm_ops.append(self.CommunicationOp(rank, e.stream, e.op_count, e.function, e.timestamp, tp_group, pp_stage, dp_group, e.group_hash))

        # Helper: keep latest op per rank
        def _reduce_latest_by_rank(ops: List[DistributedLogAnalyzer.CommunicationOp]) -> List[DistributedLogAnalyzer.CommunicationOp]:
            latest_by_rank: Dict[int, DistributedLogAnalyzer.CommunicationOp] = {}
            for op in ops:
                ex = latest_by_rank.get(op.rank)
                if (ex is None) or (op.timestamp > ex.timestamp):
                    latest_by_rank[op.rank] = op
            return list(latest_by_rank.values())

        # 1) GLOBAL groups: groupHash present across all ranks
        ops_by_hash: Dict[str, List[DistributedLogAnalyzer.CommunicationOp]] = defaultdict(list)
        ranks_by_hash: Dict[str, set] = defaultdict(set)
        for op in comm_ops:
            if op.group_hash:
                ops_by_hash[op.group_hash].append(op)
                ranks_by_hash[op.group_hash].add(op.rank)

        global_hashes = {gh for gh, rset in ranks_by_hash.items() if len(rset) == world_size}

        # Build groups dict: include GLOBAL and per-type TP/PP/DP based on group ids and group_hash
        groups: Dict[str, Dict[Tuple[int|str,int,str], List[DistributedLogAnalyzer.CommunicationOp]]] = {
            'GLOBAL': defaultdict(list), 'TP': defaultdict(list), 'PP': defaultdict(list), 'DP': defaultdict(list)
        }

        # Populate GLOBAL by ("GLOBAL", opCount, group_hash)
        for gh in global_hashes:
            for op in ops_by_hash[gh]:
                groups['GLOBAL'][("GLOBAL", op.op_count, gh)].append(op)

        # Populate TP/PP/DP by (group_id, opCount, group_hash)
        for op in comm_ops:
            if not op.group_hash:
                continue
            groups['TP'][(op.tp_group, op.op_count, op.group_hash)].append(op)
            groups['DP'][(op.dp_group, op.op_count, op.group_hash)].append(op)

        # Temporarily disable PP analysis (to be implemented later)
        groups['PP'].clear()
        
        def _reduce_group_ops_by_rank_latest(ops: List[DistributedLogAnalyzer.CommunicationOp]) -> List[DistributedLogAnalyzer.CommunicationOp]:
            latest_by_rank = {}
            for op in ops:
                existing = latest_by_rank.get(op.rank)
                if (existing is None) or (op.timestamp > existing.timestamp):
                    latest_by_rank[op.rank] = op
            return list(latest_by_rank.values())
        # Build expected full membership per group id (based on positions)
        expected_tp_members: Dict[int, set] = defaultdict(set)
        expected_dp_members: Dict[int, set] = defaultdict(set)
        all_ranks = set(op.rank for op in comm_ops)
        for op in comm_ops:
            expected_tp_members[op.tp_group].add(op.rank)
            expected_dp_members[op.dp_group].add(op.rank)

        # Group stats for diagnostics
        try:
            for ct in ['GLOBAL', 'TP', 'PP', 'DP']:
                sizes = []
                for _, ops in groups[ct].items():
                    reduced = _reduce_latest_by_rank(ops)
                    sizes.append(len(reduced))
                total = len(sizes)
                usable = sum(1 for s in sizes if s > 1)
                max_sz = max(sizes) if sizes else 0
                self.logger.info(
                    f"[ParallelSlow] {ct} group_count={total} usable(>1)={usable} max_group_size={max_sz}"
                )
        except Exception:
            pass
        # Slow counts and cumulative lag durations (seconds) by rank and comm type
        scores = defaultdict(lambda: {'GLOBAL': 0, 'TP': 0, 'PP': 0, 'DP': 0, 'BARRIER': 0})
        durations = defaultdict(lambda: {'GLOBAL': 0.0, 'TP': 0.0, 'PP': 0.0, 'DP': 0.0, 'BARRIER': 0.0})
        participations = defaultdict(lambda: {'GLOBAL': 0, 'TP': 0, 'PP': 0, 'DP': 0, 'BARRIER': 0})
        # Accumulate raw slow counts per-group for per-group normalization
        per_group_counts: Dict[str, Dict[object, Dict[int, int]]] = {
            'GLOBAL': defaultdict(lambda: defaultdict(int)),
            'TP': defaultdict(lambda: defaultdict(int)),
            'DP': defaultdict(lambda: defaultdict(int)),
            'PP': defaultdict(lambda: defaultdict(int)),
            'BARRIER': defaultdict(lambda: defaultdict(int)),
        }
        total_groups_analyzed = 0
        total_slow_picks = 0
        for comm_type, type_groups in groups.items():
            groups_analyzed = 0
            slow_picks = 0
            for group_key, ops in type_groups.items():
                reduced_ops = _reduce_latest_by_rank(ops)
                # Enforce completeness: comparison valid only if all expected members present
                if comm_type == 'GLOBAL':
                    expected = all_ranks
                elif comm_type == 'TP':
                    expected = expected_tp_members.get(group_key[0], set())
                elif comm_type == 'DP':
                    expected = expected_dp_members.get(group_key[0], set())
                else:
                    expected = set()
                present = set(op.rank for op in reduced_ops)
                if expected and (present != expected):
                    continue
                if len(reduced_ops) > 1:
                    # Count participations for denominator (how many comparisons this rank was involved in)
                    for op in reduced_ops:
                        participations[op.rank][comm_type] += 1
                    # Pick slowest and second slowest
                    reduced_ops.sort(key=lambda x: x.timestamp)
                    slowest = reduced_ops[-1]
                    second = reduced_ops[-2]
                    delta = max(0.0, float(slowest.timestamp - second.timestamp))
                    groups_analyzed += 1
                    scores[slowest.rank][comm_type] += 1
                    # track per-group raw counts (group id for TP/DP, groupHash for GLOBAL)
                    group_id_key: object
                    if comm_type == 'TP' or comm_type == 'DP' or comm_type == 'PP' or comm_type == 'BARRIER':
                        group_id_key = group_key[0]
                    else:  # GLOBAL
                        group_id_key = group_key[2]  # use groupHash as group id
                    per_group_counts[comm_type][group_id_key][slowest.rank] += 1
                    durations[slowest.rank][comm_type] += delta
                    slow_picks += 1
                    try:
                        self.logger.info(
                            f"[ParallelSlow] PICK {comm_type} group_id={group_key[0]} opCount={group_key[1]} gh={group_key[2]} "
                            f"slowest_rank={slowest.rank} ts={slowest.timestamp:.6f} second_ts={second.timestamp:.6f} delta={delta:.6f}"
                        )
                    except Exception:
                        pass
            total_groups_analyzed += groups_analyzed
            total_slow_picks += slow_picks
        # Normalize counts per group id within each communication type (subtract min per group)
        normalized_scores: Dict[int, Dict[str, int]] = {r: {'GLOBAL':0,'TP':0,'PP':0,'DP':0,'BARRIER':0} for r in scores}
        for comm_type in ['GLOBAL','TP','PP','DP']:
            for gid, rank_map in per_group_counts[comm_type].items():
                if not rank_map:
                    continue
                m = min(rank_map.values())
                for r, c in rank_map.items():
                    normalized_scores.setdefault(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0,'BARRIER':0})
                    normalized_scores[r][comm_type] += (c - m)
        return {
            'configuration': {'tp_size': tp_size, 'pp_size': pp_size, 'dp_size': dp_size, 'world_size': world_size},
            'analysis_summary': {
                'total_operations': sum(len(v) for v in rank_entries.values()),
                'global_groups': len(groups['GLOBAL']), 'tp_groups': len(groups['TP']), 'pp_groups': len(groups['PP']), 'dp_groups': len(groups['DP'])
            },
            'raw_scores': dict(scores),
            'slow_durations': dict(durations),
            'participations': dict(participations),
            'normalized_scores': normalized_scores,
            'total_groups_analyzed': total_groups_analyzed,
            'total_slow_picks': total_slow_picks
        }
    
    def analyze_rank_parallel_positions(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze rank parallel positions for each node when model training parameters are provided
        
        Args:
            model_config: Model training configuration parameters
            
        Returns:
            Rank parallel position analysis results
        """
        self.logger.info("Starting rank parallel position analysis...")
        
        # Extract model parameters
        model_size = model_config.get('model_size', 0)
        num_layers = model_config.get('num_layers', 0)
        hidden_size = model_config.get('hidden_size', 0)
        num_attention_heads = model_config.get('num_attention_heads', 0)
        vocab_size = model_config.get('vocab_size', 0)
        max_position_embeddings = model_config.get('max_position_embeddings', 0)
        
        # Get rank information from analysis results
        total_ranks = self.analysis_results.get('total_ranks', 0)
        last_operations = self.analysis_results.get('last_operations', {})
        
        if not last_operations:
            self.logger.error("No rank information available for parallel position analysis")
            return {}
        
        # Analyze parallel positions for each rank
        rank_parallel_positions = {}
        for rank in sorted(last_operations.keys()):
            # Calculate parallel position based on rank
            parallel_info = self._calculate_parallel_position(
                rank, total_ranks, model_size, num_layers, hidden_size, 
                num_attention_heads, vocab_size, max_position_embeddings
            )
            rank_parallel_positions[rank] = parallel_info
        
        # Generate configuration file
        config_data = self._generate_parallel_config(
            rank_parallel_positions, model_config, total_ranks
        )
        
        # Save config to file
        config_file = Path('config.yaml')
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            self.logger.info(f"Parallel configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration file: {e}")
        
        # Update analysis results
        self.analysis_results['rank_parallel_positions'] = rank_parallel_positions
        self.analysis_results['parallel_config_file'] = str(config_file)
        
        return rank_parallel_positions
    
    def _calculate_parallel_position(self, rank: int, total_ranks: int, 
                                   model_size: int, num_layers: int, hidden_size: int,
                                   num_attention_heads: int, vocab_size: int, 
                                   max_position_embeddings: int) -> Dict[str, Any]:
        """
        Calculate parallel position for a specific rank
        
        Args:
            rank: Rank number
            total_ranks: Total number of ranks
            model_size: Model size in parameters
            num_layers: Number of layers
            hidden_size: Hidden size
            num_attention_heads: Number of attention heads
            vocab_size: Vocabulary size
            max_position_embeddings: Maximum position embeddings
            
        Returns:
            Parallel position information
        """
        parallel_info = {
            'rank': rank,
            'total_ranks': total_ranks,
            'parallel_strategy': 'tensor_parallel' if total_ranks > 1 else 'single_gpu',
            'layer_distribution': {},
            'parameter_distribution': {},
            'memory_distribution': {},
            'parallel_communication': {}
        }
        
        if total_ranks > 1:
            # Calculate layer distribution for parallel processing
            layers_per_rank = num_layers // total_ranks
            start_layer = rank * layers_per_rank
            end_layer = start_layer + layers_per_rank if rank < total_ranks - 1 else num_layers
            
            parallel_info['layer_distribution'] = {
                'start_layer': start_layer,
                'end_layer': end_layer,
                'layers_per_rank': end_layer - start_layer,
                'parallel_workload': (end_layer - start_layer) / num_layers * 100
            }
            
            # Calculate parameter distribution for parallel computation
            if model_size > 0:
                params_per_rank = model_size // total_ranks
                start_param = rank * params_per_rank
                end_param = start_param + params_per_rank if rank < total_ranks - 1 else model_size
                
                parallel_info['parameter_distribution'] = {
                    'start_param': start_param,
                    'end_param': end_param,
                    'params_per_rank': end_param - start_param,
                    'parallel_percentage': ((end_param - start_param) / model_size) * 100
                }
            
            # Calculate memory distribution for parallel execution
            if hidden_size > 0 and num_layers > 0:
                # Estimate memory usage based on parallel model architecture
                attention_memory = (hidden_size * hidden_size * 4) // total_ranks  # 4 bytes per float
                ffn_memory = (hidden_size * hidden_size * 4 * 4) // total_ranks  # 4 bytes per float, 4 for FFN
                embedding_memory = (vocab_size * hidden_size * 4) // total_ranks if rank == 0 else 0
                
                parallel_info['memory_distribution'] = {
                    'attention_memory_mb': attention_memory / (1024 * 1024),
                    'ffn_memory_mb': ffn_memory / (1024 * 1024),
                    'embedding_memory_mb': embedding_memory / (1024 * 1024),
                    'total_memory_mb': (attention_memory + ffn_memory + embedding_memory) / (1024 * 1024),
                    'parallel_memory_efficiency': total_ranks  # Memory efficiency factor
                }
            
            # Calculate parallel communication patterns
            parallel_info['parallel_communication'] = {
                'allreduce_operations': num_layers,  # Number of allreduce operations per layer
                'communication_volume': (hidden_size * hidden_size * 4) // total_ranks,  # Communication volume per operation
                'parallel_efficiency': 1.0 / total_ranks,  # Theoretical parallel efficiency
                'communication_overhead': total_ranks - 1  # Communication overhead factor
            }
        
        return parallel_info
    
    def _generate_parallel_config(self, rank_parallel_positions: Dict[int, Dict[str, Any]], 
                                model_config: Dict[str, Any], total_ranks: int) -> Dict[str, Any]:
        """
        Generate parallel configuration file
        
        Args:
            rank_parallel_positions: Rank parallel position analysis results
            model_config: Model training configuration
            total_ranks: Total number of ranks
            
        Returns:
            Configuration data for YAML file
        """
        config_data = {
            'model_configuration': {
                'model_size': model_config.get('model_size', 0),
                'num_layers': model_config.get('num_layers', 0),
                'hidden_size': model_config.get('hidden_size', 0),
                'num_attention_heads': model_config.get('num_attention_heads', 0),
                'vocab_size': model_config.get('vocab_size', 0),
                'max_position_embeddings': model_config.get('max_position_embeddings', 0)
            },
            'parallel_training': {
                'total_ranks': total_ranks,
                'parallel_strategy': 'tensor_parallel' if total_ranks > 1 else 'single_gpu',
                'rank_parallel_configuration': {}
            }
        }
        
        # Add rank-specific parallel configuration
        for rank, parallel_info in rank_parallel_positions.items():
            config_data['parallel_training']['rank_parallel_configuration'][f'rank_{rank}'] = {
                'rank_id': rank,
                'layer_distribution': parallel_info['layer_distribution'],
                'parameter_distribution': parallel_info['parameter_distribution'],
                'memory_distribution': parallel_info['memory_distribution'],
                'parallel_communication': parallel_info['parallel_communication']
            }
        
        return config_data
    
    def load_config_and_analyze(self, config_file_path: str = 'config.yaml') -> Dict[str, Any]:
        """
        Load existing config.yaml file and analyze rank parallel positions
        
        Args:
            config_file_path: Path to the existing config.yaml file
            
        Returns:
            Rank parallel position analysis results based on loaded config
        """
        self.logger.info(f"Loading configuration from {config_file_path}")
        
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            self.logger.info("Configuration file loaded successfully")
            
            # Extract model configuration
            model_config = config_data.get('model_configuration', {})
            if not model_config:
                self.logger.error("No model configuration found in config file")
                return {}
            
            # Extract parallel training configuration
            parallel_config = config_data.get('parallel_training', {})
            total_ranks = parallel_config.get('total_ranks', 0)
            
            if total_ranks == 0:
                self.logger.error("No total_ranks found in parallel training configuration")
                return {}
            
            self.logger.info(f"Model configuration: {model_config}")
            self.logger.info(f"Parallel training configuration: {parallel_config}")
            
            # Analyze rank parallel positions based on loaded config
            rank_parallel_positions = self.analyze_rank_parallel_positions(model_config)
            
            # Validate analysis results against config
            self._validate_analysis_against_config(rank_parallel_positions, parallel_config)
            
            return rank_parallel_positions
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file {config_file_path} not found")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _validate_analysis_against_config(self, rank_parallel_positions: Dict[str, Any], 
                                        parallel_config: Dict[str, Any]) -> None:
        """
        Validate analysis results against loaded configuration
        
        Args:
            rank_parallel_positions: Analysis results
            parallel_config: Loaded parallel configuration
        """
        self.logger.info("Validating analysis results against configuration...")
        
        expected_ranks = parallel_config.get('total_ranks', 0)
        actual_ranks = len(rank_parallel_positions)
        
        if expected_ranks != actual_ranks:
            self.logger.warning(f"Rank count mismatch: expected {expected_ranks}, got {actual_ranks}")
        
        # Check if rank configuration exists in config
        rank_config = parallel_config.get('rank_parallel_configuration', {})
        if rank_config:
            self.logger.info("Found existing rank configuration in config file")
            
            # Compare with analysis results
            for rank_id, config_info in rank_config.items():
                rank_num = config_info.get('rank_id', 0)
                if rank_num in rank_parallel_positions:
                    analysis_info = rank_parallel_positions[rank_num]
                    
                    # Compare layer distribution
                    config_layers = config_info.get('layer_distribution', {})
                    analysis_layers = analysis_info.get('layer_distribution', {})
                    
                    if config_layers and analysis_layers:
                        if (config_layers.get('start_layer') != analysis_layers.get('start_layer') or
                            config_layers.get('end_layer') != analysis_layers.get('end_layer')):
                            self.logger.warning(f"Layer distribution mismatch for rank {rank_num}")
                    
                    # Compare parameter distribution
                    config_params = config_info.get('parameter_distribution', {})
                    analysis_params = analysis_info.get('parameter_distribution', {})
                    
                    if config_params and analysis_params:
                        if (config_params.get('start_param') != analysis_params.get('start_param') or
                            config_params.get('end_param') != analysis_params.get('end_param')):
                            self.logger.warning(f"Parameter distribution mismatch for rank {rank_num}")
        
        self.logger.info("Configuration validation completed")
    
    def analyze_3d_parallel_ranks(self, tp_size: int, pp_size: int, n_workers: int) -> Dict[str, Any]:
        """
        Analyze 3D parallel (TP, PP, DP) rank distribution based on C++ logic
        
        Args:
            tp_size: Tensor Parallel size
            pp_size: Pipeline Parallel size
            n_workers: Total number of workers/ranks
            
        Returns:
            3D parallel rank analysis results
        """
        self.logger.info(f"Starting 3D parallel rank analysis: TP={tp_size}, PP={pp_size}, N_WORKERS={n_workers}")
        
        # Calculate DP size based on C++ logic
        dp_size = n_workers // (tp_size * pp_size)
        
        if dp_size == 0:
            self.logger.error("Invalid parallel configuration: dp_size cannot be 0")
            return {}
        
        # Calculate group sizes
        tp_group_size = n_workers // tp_size
        pp_group_size = n_workers // pp_size
        dp_group_size = n_workers // dp_size
        
        self.logger.info(f"Calculated sizes: DP_SIZE={dp_size}, TP_GROUP_SIZE={tp_group_size}, PP_GROUP_SIZE={pp_group_size}, DP_GROUP_SIZE={dp_group_size}")
        
        # Initialize groups
        tp_groups = [[] for _ in range(tp_group_size)]
        pp_groups = [[] for _ in range(pp_group_size)]
        dp_groups = [[] for _ in range(dp_group_size)]
        
        # Initialize rank information
        ranks_info = {}
        
        # Initialize each rank based on C++ logic
        for i in range(n_workers):
            # Calculate PP stage based on C++ logic: i / (N_WORKERS / PP_SIZE)
            # PP represents the pipeline parallel stage (0, 1, 2, ...)
            pp_stage = i // (n_workers // pp_size)
            
            rank_info = {
                'rank_id': i,
                'n_workers': n_workers,
                'tp': i % tp_size,  # TP = i % TP_SIZE (Tensor Parallel position within group)
                'tp_group': i // tp_size,  # TP_GROUP = i // TP_SIZE (Tensor Parallel group number)
                'pp': pp_stage,  # PP stage (0, 1, 2, ...)
                'pp_group': i % (n_workers // pp_size),  # PP_GROUP = i % (N_WORKERS // PP_SIZE) (Pipeline Parallel group number)
                'dp': (i // tp_size) % dp_size,  # DP = (i // TP_SIZE) % DP_SIZE (Data Parallel position within group)
                'dp_group': (i // (tp_size * dp_size)) * tp_size + i % tp_size,  # DP_GROUP calculation
                'is_first_pp': pp_stage == 0,  # First PP stage
                'is_last_pp': pp_stage == pp_size - 1,  # Last PP stage
                'parallel_type': '3D_parallel'
            }
            
            # Add to groups
            tp_groups[rank_info['tp_group']].append(i)
            pp_groups[rank_info['pp_group']].append(i)
            dp_groups[rank_info['dp_group']].append(i)
            
            ranks_info[i] = rank_info
        
        # Generate 3D parallel configuration
        config_data = self._generate_3d_parallel_config(
            ranks_info, tp_groups, pp_groups, dp_groups,
            tp_size, pp_size, dp_size, n_workers
        )
        
        # Save 3D parallel config to file
        config_file = Path('3d_parallel_config.yaml')
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            self.logger.info(f"3D parallel configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save 3D parallel configuration file: {e}")
        
        # Update analysis results
        self.analysis_results['3d_parallel_ranks'] = ranks_info
        self.analysis_results['3d_parallel_groups'] = {
            'tp_groups': tp_groups,
            'pp_groups': pp_groups,
            'dp_groups': dp_groups
        }
        self.analysis_results['3d_parallel_config_file'] = str(config_file)
        
        return ranks_info
    
    def _generate_3d_parallel_config(self, ranks_info: Dict[int, Dict[str, Any]], 
                                   tp_groups: List[List[int]], pp_groups: List[List[int]], 
                                   dp_groups: List[List[int]], tp_size: int, pp_size: int, 
                                   dp_size: int, n_workers: int) -> Dict[str, Any]:
        """
        Generate 3D parallel configuration file
        
        Args:
            ranks_info: Rank information dictionary
            tp_groups: Tensor parallel groups
            pp_groups: Pipeline parallel groups
            dp_groups: Data parallel groups
            tp_size: Tensor parallel size
            pp_size: Pipeline parallel size
            dp_size: Data parallel size
            n_workers: Total number of workers
            
        Returns:
            3D parallel configuration data
        """
        config_data = {
            '3d_parallel_configuration': {
                'parallel_sizes': {
                    'tp_size': tp_size,
                    'pp_size': pp_size,
                    'dp_size': dp_size,
                    'n_workers': n_workers
                },
                'group_sizes': {
                    'tp_group_size': len(tp_groups),
                    'pp_group_size': len(pp_groups),
                    'dp_group_size': len(dp_groups)
                },
                'rank_configuration': {}
            }
        }
        
        # Add rank-specific configuration
        for rank_id, rank_info in ranks_info.items():
            config_data['3d_parallel_configuration']['rank_configuration'][f'rank_{rank_id}'] = {
                'rank_id': rank_id,
                'tp': rank_info['tp'],
                'tp_group': rank_info['tp_group'],
                'pp': rank_info['pp'],
                'pp_group': rank_info['pp_group'],
                'dp': rank_info['dp'],
                'dp_group': rank_info['dp_group'],
                'is_first_pp': rank_info['is_first_pp'],
                'is_last_pp': rank_info['is_last_pp'],
                'parallel_type': rank_info['parallel_type']
            }
        
        # Add group information
        config_data['3d_parallel_configuration']['groups'] = {
            'tp_groups': {f'group_{i}': group for i, group in enumerate(tp_groups)},
            'pp_groups': {f'group_{i}': group for i, group in enumerate(pp_groups)},
            'dp_groups': {f'group_{i}': group for i, group in enumerate(dp_groups)}
        }
        
        return config_data
    
    def print_3d_parallel_summary(self, ranks_info: Dict[int, Dict[str, Any]], 
                                tp_groups: List[List[int]], pp_groups: List[List[int]], 
                                dp_groups: List[List[int]]) -> None:
        """
        Print 3D parallel rank summary similar to C++ output
        
        Args:
            ranks_info: Rank information dictionary
            tp_groups: Tensor parallel groups
            pp_groups: Pipeline parallel groups
            dp_groups: Data parallel groups
        """
        print("\n" + "="*60)
        print("3D Parallel Rank Analysis Summary")
        print("="*60)
        
        # Print individual rank information
        print("\nIndividual Rank Information:")
        for rank_id in sorted(ranks_info.keys()):
            rank_info = ranks_info[rank_id]
            print(f"Rank {rank_id} initialized: "  # 直接使用rank_id，不从1开始
                  f"TP: {rank_info['tp']} "
                  f"PP: {rank_info['pp']} "
                  f"DP: {rank_info['dp']}")
        
        # Print TP groups
        print("\nTP Groups:")
        for i, group in enumerate(tp_groups):
            print(f"  group [ {i} ]: {' '.join(map(str, group))}")
        
        # Print PP groups
        print("\nPP Groups:")
        for i, group in enumerate(pp_groups):
            print(f"  group [ {i} ]: {' '.join(map(str, group))}")
        
        # Print DP groups
        print("\nDP Groups:")
        for i, group in enumerate(dp_groups):
            print(f"  group [ {i} ]: {' '.join(map(str, group))}")
        
        print("\n" + "="*60)
    
    def run(self):
        """Run complete hang detection analysis"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Distributed Training Log Hang Detection Analysis")
        self.logger.info("=" * 60)
        
        try:
            # 1. Traverse log files in folder
            self.discover_log_files()
            if not self.log_files:
                self.logger.error("No log files found for analysis")
            return
            
            # 2. Read logs for each rank and mark rank numbers
            # 3. Analyze last save_count group log entries
            rank_entries = self.parse_log_files()
            if not rank_entries:
                self.logger.error("No valid log entries parsed")
                return
            
            # Get last save_count group entries
            last_save_count_entries = self.get_last_save_count_group(rank_entries)
            
            # 4. Group logs by stream
            stream_groups = self.group_by_stream(last_save_count_entries)
            
            # 5. Identify last operation in each stream
            last_operations = self.find_last_operation_in_streams(stream_groups)
            
            # Detect hang situations
            hangs = self.detect_hangs(last_operations, stream_groups)
            
            # Output analysis results
            self.logger.info("=" * 60)
            self.logger.info("Hang Detection Analysis Completed")
            self.logger.info("=" * 60)
            
            if hangs:
                self.logger.info(f"Hang detection completed: found {len(hangs)} potential hang situations")
            else:
                self.logger.info("No hang situations detected, system running normally")
            
            # Perform time pattern analysis if available
            time_pattern_results = None
            if TimePatternAnalyzer is not None:
                try:
                    self.logger.info("Starting time pattern analysis...")
                    time_analyzer = TimePatternAnalyzer(verbose=self.verbose)
                    time_pattern_results = time_analyzer.analyze_time_patterns(rank_entries)
                    
                    # Print time pattern analysis results
                    time_analyzer.print_analysis_results(time_pattern_results)
                    
                    # Generate detailed report
                    time_analyzer.generate_time_pattern_report(time_pattern_results)
                    
                except Exception as e:
                    self.logger.error(f"Error during time pattern analysis: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
            
            # Save analysis results
            self.analysis_results = {
                'total_ranks': len(rank_entries),
                'total_streams': sum(len(streams) for streams in stream_groups.values()),
                'hangs_detected': len(hangs),
                'hang_details': hangs,
                'last_operations': last_operations,
                'time_pattern_analysis': time_pattern_results
            }
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Distributed Training Log Hang Detection Analyzer')
    parser.add_argument('log_path', help='Path to log file or directory containing log files to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = DistributedLogAnalyzer(args.log_path, args.verbose)
    analyzer.run()


if __name__ == '__main__':
    main() 