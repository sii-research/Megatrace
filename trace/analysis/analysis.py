#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Training Log Analyzer - Analyze all log files in specified path
Support hang analysis and slow analysis functionality
"""

import traceback
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Import modules with fallback for different execution contexts
try:
    from trace.analysis.group_hash_slow_detector import GroupHashSlowDetector
    from trace.analysis.log_reader import LogEntry, LogReader
except ImportError:
    # Fallback: import from same directory when running as script
    try:
        from group_hash_slow_detector import GroupHashSlowDetector
        from log_reader import LogEntry, LogReader
    except ImportError:
        # Last resort: try relative imports
        from .group_hash_slow_detector import GroupHashSlowDetector
        from .log_reader import LogEntry, LogReader

# Import time pattern analyzer
try:
    from time_pattern_analyzer import TimePatternAnalyzer
except ImportError:
    TimePatternAnalyzer = None

class DistributedLogAnalyzer:
    """Distributed Training Log Analyzer Main Class"""
    
    def __init__(self, log_path: str = None,
                 verbose: bool = False, max_save_count_groups: Optional[int] = 2):
        """
        Initialize Distributed Log Analyzer
        
        Args:
            log_path: Path to log file or directory to analyze
            verbose: Whether to show detailed log output
            max_save_count_groups: Number of most recent save_count groups to load from local files
                                   (None or <=0 means load entire file)
        """
        self.log_path = Path(log_path) if log_path else None
        self.analysis_results = {}
        self.verbose = verbose
        self.max_save_count_groups = max_save_count_groups if (max_save_count_groups is None or max_save_count_groups >= 0) else None
        # Setup logging
        self._setup_logging()
        
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

    def  _get_rank_metadata(self, rank_entries: Dict[int, List[LogEntry]], rank: int) -> Dict[str, str]:
        """
        Get metadata (node_ip, hostname, gpu_pci) for a specific rank.
        
        Args:
            rank_entries: Dictionary of rank to log entries
            rank: Rank number
            
        Returns:
            Dictionary containing node_ip, hostname, and gpu_pci
        """
        if rank not in rank_entries or not rank_entries[rank]:
            return {'node_ip': 'unknown', 'hostname': 'unknown', 'gpu_pci': 'unknown'}
        
        # Get metadata from the first entry (all entries for the same rank should have same metadata)
        first_entry = rank_entries[rank][0]
        return {
            'node_ip': first_entry.node_ip,
            'hostname': first_entry.hostname,
            'gpu_pci': first_entry.gpu_pci
        }
    
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
    
    def run_hang(self, rank_entries: Dict[int, List[LogEntry]], analyze_opcount: bool = False):
        """
        Run complete hang detection analysis on pre-loaded rank entries.
        
        Args:
            rank_entries: Dictionary of rank to log entries
            analyze_opcount: If True, perform opCount synchronization analysis (default: False)
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Distributed Training Log Hang Detection Analysis")
        self.logger.info("=" * 80)
        
        try:
            if not rank_entries:
                self.logger.error("No valid log entries parsed")
                return
            
            # Get last save_count group entries
            # NOTE: rank_entries already contains only the max save_count entries, so no need to filter
            # last_save_count_entries = self.get_last_save_count_group(rank_entries)
            last_save_count_entries = rank_entries
            
            # 4. Group logs by stream
            stream_groups = self.group_by_stream(last_save_count_entries)
            
            # 5. Identify last operation in each stream
            last_operations = self.find_last_operation_in_streams(stream_groups)
            
            # Detect hang situations
            hangs = self.detect_hangs(last_operations, stream_groups)
            
            # Output analysis results
            self.logger.info("=" * 80)
            self.logger.info("Hang Detection Analysis Completed")
            self.logger.info("=" * 80)
            
            if hangs:
                self.logger.info(f"Hang detection completed: found {len(hangs)} potential hang situations")
            else:
                self.logger.info("No hang situations detected, system running normally")
            
            # Perform opCount synchronization analysis if requested
            opcount_analysis = None
            if analyze_opcount:
                opcount_analysis = self._analyze_opcount_synchronization(last_operations, stream_groups)
            
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
            
            if opcount_analysis:
                self.analysis_results['opcount_sync_analysis'] = opcount_analysis
            
            # Find and output min op count rank (the rank that is stuck/causing hang)
            min_opcount_rank = self._find_min_opcount_rank_from_last_ops(last_operations)
            if min_opcount_rank is not None:
                rank_metadata = self._get_rank_metadata(rank_entries, min_opcount_rank)
                result = "Min OpCount Rank (Stuck Rank) Information:\n"
                result += f"Rank: {min_opcount_rank}, PCI: {rank_metadata['gpu_pci']}, \n"
                result += f"Hostname: {rank_metadata['hostname']}, IP: {rank_metadata['node_ip']}\n"
                result += ("=" * 80)
                result += '\n'
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info("Min OpCount Rank (Stuck Rank) Information:")
                self.logger.info(f"Rank: {min_opcount_rank}, PCI: {rank_metadata['gpu_pci']}")
                self.logger.info(f"Hostname: {rank_metadata['hostname']}, IP: {rank_metadata['node_ip']}")
                self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            traceback.print_exc()
            return None
    
    def _find_min_opcount_rank_from_last_ops(self, last_operations: Dict[int, Dict[str, LogEntry]]) -> Optional[int]:
        """
        Find the rank with minimum op_count from last operations (the rank that is stuck/causing hang).
        This is a simplified version that finds the minimum op_count across all ranks and streams.
        
        Args:
            last_operations: Last operations by rank and stream
            
        Returns:
            Rank number with minimum op_count, or None if no operations found
        """
        if not last_operations:
            return None
        
        # Build opCount map to find unique opCounts
        opcount_map = {}
        for rank, sdict in last_operations.items():
            for stream, op in sdict.items():
                opcount_map.setdefault(op.op_count, []).append((rank, op.function, stream))
        
        # Find unique opCounts (appearing only once)
        unique_items = [(oc, vals[0]) for oc, vals in sorted(opcount_map.items()) if len(vals) == 1]
        
        if unique_items:
            # Find minimum unique opCount
            min_oc, (min_rank, _, _) = min(unique_items, key=lambda x: x[0])
            return min_rank
        else:
            # If no unique opCounts, find the minimum opCount overall
            min_opcount = None
            min_opcount_rank = None
            for rank, sdict in last_operations.items():
                for stream, op in sdict.items():
                    if min_opcount is None or op.op_count < min_opcount:
                        min_opcount = op.op_count
                        min_opcount_rank = rank
            return min_opcount_rank
    
    def _analyze_opcount_synchronization(self, last_operations: Dict[int, Dict[str, LogEntry]], 
                                        stream_groups: Dict[int, Dict[str, List[LogEntry]]]) -> Dict:
        """
        Perform opCount synchronization analysis.
        
        Args:
            last_operations: Last operations by rank and stream
            stream_groups: Stream groups by rank
            
        Returns:
            Dictionary containing opCount synchronization analysis results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Summary Analysis - Synchronization Comparison:")
        self.logger.info("=" * 80)
        
        unique_ranks = len(last_operations)
        total_streams = sum(len(streams) for streams in stream_groups.values())
        self.logger.info(f"Total Ranks: {unique_ranks}")
        self.logger.info(f"Total Streams: {total_streams}")
        
        # Log last operations by rank and stream
        self.logger.info("\nLast Operations by Rank and Stream:")
        for rank in sorted(last_operations.keys()):
            self.logger.info(f"Rank {rank}:")
            for stream in sorted(last_operations[rank].keys()):
                op = last_operations[rank][stream]
                self.logger.info(f"  Stream {stream}: {op.function} (opCount {op.op_count})")
        
        # Build opCount map
        opcount_map = {}
        for rank, sdict in last_operations.items():
            for stream, op in sdict.items():
                opcount_map.setdefault(op.op_count, []).append((rank, op.function, stream))
        
        duplicate_items = [(oc, vals) for oc, vals in sorted(opcount_map.items()) if len(vals) > 1]
        unique_items = [(oc, vals[0]) for oc, vals in sorted(opcount_map.items()) if len(vals) == 1]
        
        opcount_analysis = {
            'duplicate_opcounts': [],
            'unique_opcounts': [],
            'min_unique_opcount': None,
            'min_unique_rank': None
        }
        
        if duplicate_items:
            self.logger.info("\nDuplicate opCounts (appearing in multiple ranks):")
            for oc, vals in duplicate_items:
                ranks = sorted(r for r, _, _ in vals)
                self.logger.info(f"  opCount {oc}: Ranks {ranks}")
                op_type_ranks = {}
                for r, func, _ in vals:
                    op_type_ranks.setdefault(func, []).append(r)
                for func in sorted(op_type_ranks.keys()):
                    self.logger.info(f"    {func}: Ranks {sorted(set(op_type_ranks[func]))}")
                
                opcount_analysis['duplicate_opcounts'].append({
                    'op_count': oc,
                    'ranks': ranks,
                    'functions': {func: sorted(set(rank_list)) for func, rank_list in op_type_ranks.items()}
                })
        
        if unique_items:
            self.logger.info("\nUnique opCounts (appearing only once):")
            for oc, (r, _, _) in unique_items:
                self.logger.info(f"  opCount {oc}: Rank {r}")
                opcount_analysis['unique_opcounts'].append({
                    'op_count': oc,
                    'rank': r
                })
            
            if unique_items:
                min_oc, (min_rank, _, _) = min(unique_items, key=lambda x: x[0])
                opcount_analysis['min_unique_opcount'] = min_oc
                opcount_analysis['min_unique_rank'] = min_rank
                self.logger.info(f"\nMinimum unique opCount: {min_oc} (Rank {min_rank})")
                
                for stream, op in stream_groups[min_rank].items():
                    lop = last_operations[min_rank].get(stream)
                    if lop and lop.op_count == min_oc:
                        self.logger.info(f"\nDetails for Rank {min_rank} (minimum unique opCount):")
                        self.logger.info(f"  Stream {stream}: {lop.function} (opCount {lop.op_count})")
                        break
        
        return opcount_analysis
    
    def run_slow(self, rank_entries: Dict[int, List[LogEntry]], config_path: Optional[str] = None, 
                 analyze_grouphash: bool = True) -> Dict:
        """
        Run slow node detection analysis on pre-loaded rank entries.
        Only performs GroupHash-based slow detection.
        
        Args:
            rank_entries: Dictionary of rank to log entries
            config_path: Path to parallel analysis config file (optional, ignored)
            analyze_grouphash: If True, perform GroupHash-based slow detection (default: True)
            
        Returns:
            Dictionary containing slow analysis results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Slow Node Detection Analysis")
        self.logger.info("=" * 80)
        
        try:
            if not rank_entries:
                self.logger.error("No valid log entries parsed")
                return {}
            
            # Perform GroupHash-based slow detection
            grouphash_results = None
            if analyze_grouphash:
                grouphash_results = self._analyze_grouphash_slow(rank_entries)
            
            # Save analysis results
            slow_analysis_results = {}
            
            if grouphash_results:
                slow_analysis_results['grouphash_analysis'] = grouphash_results
            
            # Store in analysis_results if it exists, otherwise create it
            if not hasattr(self, 'analysis_results') or not self.analysis_results:
                self.analysis_results = {}
            self.analysis_results['slow_analysis'] = slow_analysis_results
            
            return slow_analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during slow analysis: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {}
    
    def _analyze_per_rank_slow(self, last_group_entries: Dict[int, List[LogEntry]]) -> tuple:
        """Analyze slow ranks using per-rank metrics."""
        
        def compute_rank_metrics(rank, entries):
            if not entries:
                return rank, None
            # sort by timestamp
            sorted_entries = sorted(entries, key=lambda x: x.timestamp)
            start_ts = sorted_entries[0].timestamp
            end_ts = sorted_entries[-1].timestamp
            total_duration = max(0.0, end_ts - start_ts)
            # intervals
            intervals = []
            for i in range(1, len(sorted_entries)):
                intervals.append(sorted_entries[i].timestamp - sorted_entries[i - 1].timestamp)
            mean_interval = float(np.mean(intervals)) if intervals else 0.0
            std_interval = float(np.std(intervals)) if intervals else 0.0
            max_interval = float(np.max(intervals)) if intervals else 0.0
            p95_interval = float(np.percentile(intervals, 95)) if intervals else 0.0
            total_calls = len(sorted_entries)
            calls_per_second = (total_calls / total_duration) if total_duration > 0 else 0.0
            # performance score: frequency 70% + consistency 30%
            frequency_score = min(100.0, calls_per_second * 10.0)
            consistency_score = max(0.0, 100.0 - ((std_interval / mean_interval) * 100.0)) if mean_interval > 0 else 0.0
            performance_score = max(0.0, min(100.0, frequency_score * 0.7 + consistency_score * 0.3))
            return rank, {
                'performance_score': performance_score,
                'calls_per_second': calls_per_second,
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'max_interval': max_interval,
                'p95_interval': p95_interval,
                'total_calls': total_calls,
                'total_duration': total_duration,
            }
        
        max_workers = 8
        if self.verbose:
            self.logger.info(f"Running multithreaded slow analysis with {max_workers} workers (per-rank, last save_count group)...")
        
        rank_metrics = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_rank_metrics, rank, entries) for rank, entries in last_group_entries.items()]
            for fut in as_completed(futures):
                rank, metrics = fut.result()
                if metrics is not None:
                    rank_metrics[rank] = metrics
        
        if not rank_metrics:
            self.logger.info("  • No rank metrics computed")
            return {}, []
        
        # Robust, threshold-based anomaly detection
        ranks = sorted(rank_metrics.keys())
        scores_arr = np.array([rank_metrics[r]['performance_score'] for r in ranks], dtype=float)
        cps_arr = np.array([rank_metrics[r]['calls_per_second'] for r in ranks], dtype=float)
        cv_list = []
        for r in ranks:
            mi = rank_metrics[r]['mean_interval']
            si = rank_metrics[r]['std_interval']
            cv_list.append((si / mi) if mi > 0 else 0.0)
        cv_arr = np.array(cv_list, dtype=float)
        
        def median_mad(arr: np.ndarray):
            m = float(np.median(arr)) if arr.size else 0.0
            mad = float(np.median(np.abs(arr - m))) if arr.size else 0.0
            return m, mad
        
        m_s, mad_s = median_mad(scores_arr)
        m_c, mad_c = median_mad(cps_arr)
        m_v, mad_v = median_mad(cv_arr)
        p10_s = float(np.percentile(scores_arr, 10)) if scores_arr.size else 0.0
        p10_c = float(np.percentile(cps_arr, 10)) if cps_arr.size else 0.0
        p90_v = float(np.percentile(cv_arr, 90)) if cv_arr.size else 0.0
        
        use_percentile_only = (len(ranks) < 5)
        slow_ranks = []
        for idx, r in enumerate(ranks):
            s = scores_arr[idx]
            f = cps_arr[idx]
            v = cv_arr[idx]
            if use_percentile_only:
                cond_score = (s < p10_s)
                cond_cps = (f < p10_c)
                cond_cv = (v > p90_v)
            else:
                cond_score = (s < (m_s - max(3.0 * mad_s, 5.0))) or (s < p10_s)
                cond_cps = (f < (m_c - max(3.0 * mad_c, 0.1 * m_c))) or (f < p10_c)
                cond_cv = (v > (m_v + max(3.0 * mad_v, 0.2))) or (v > p90_v)
            hits = (1 if cond_score else 0) + (1 if cond_cps else 0) + (1 if cond_cv else 0)
            if hits >= 2:
                slow_ranks.append(r)
        
        self.logger.info("\nSIMPLE SLOW RANK ANALYSIS (multithreaded):")
        self.logger.info(f"  • Total Ranks Analyzed: {len(rank_metrics)}")
        if slow_ranks:
            self.logger.info(f"  • Slow Ranks Detected: {sorted(slow_ranks)}")
            for r in sorted(slow_ranks):
                rm = rank_metrics[r]
                cv_val = (cv_arr[ranks.index(r)])
                metadata = self._get_rank_metadata(last_group_entries, r)
                self.logger.info(f"    Rank {r} (node_ip={metadata['node_ip']}, hostname={metadata['hostname']}, gpu_pci={metadata['gpu_pci']}):")
                self.logger.info(f"      • Score: {rm['performance_score']:.3f}")
                self.logger.info(f"      • Calls/sec: {rm['calls_per_second']:.2f} (total_calls={rm['total_calls']}, duration={rm['total_duration']:.2f}s)")
                self.logger.info(f"      • Interval mean/std/p95/max: {rm['mean_interval']:.4f}s / {rm['std_interval']:.4f}s / {rm['p95_interval']:.4f}s / {rm['max_interval']:.4f}s")
                self.logger.info(f"      • CV: {cv_val:.3f}")
        else:
            self.logger.info("  • No obvious anomalies.")
        
        return rank_metrics, slow_ranks
    
    def _analyze_parallel_slow(self, rank_entries: Dict[int, List[LogEntry]], config_path: str) -> Optional[Dict]:
        """Analyze parallel slow nodes using config file."""
        
        try:
            cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
            tp = cfg.get('TP', 2)
            pp = cfg.get('PP', 2)
            ws = cfg.get('world_size', 8)
            
            if tp * pp > ws:
                self.logger.warning("Invalid config (TP*PP > world_size); skip parallel slow analysis")
                return None
            
            parallel = self.analyze_parallel_slow_nodes(rank_entries, tp, pp, ws)
            
            if 'error' in parallel:
                self.logger.error(f"Parallel slow analysis error: {parallel['error']}")
                return None
            
            self.logger.info("\nPARALLEL SLOW NODE ANALYSIS:")
            self.logger.info(f"  • Configuration: TP={tp}, PP={pp}, DP={ws // (tp*pp)}")
            self.logger.info(f"  • Groups Analyzed: {parallel['total_groups_analyzed']}")
            if 'total_slow_picks' in parallel:
                self.logger.info(f"  • Slow Picks: {parallel['total_slow_picks']}")
            
            # Log detailed per-rank table
            normalized = parallel.get('normalized_scores', {}) or {}
            raw = parallel.get('raw_scores', {}) or {}
            durations = parallel.get('slow_durations', {}) or {}
            parts = parallel.get('participations', {}) or {}
            rank_list = sorted(rank_entries.keys())
            
            self.logger.info("  • Normalized Slow Counts / Rate (relative to min per type; rate = raw/participations):")
            self.logger.info(f"    {'Rank':<6} {'node_ip':<16} {'hostname':<16} {'gpu_pci':<16} {'GLOBAL':<16} {'TP':<16} {'PP':<16} {'DP':<16} {'Total':<10}")
            self.logger.info("    " + "-" * 148)
            
            def pct(raw_val, part_val):
                return 0.0 if part_val == 0 else (100.0 * float(raw_val) / float(part_val))
            
            for r in rank_list:
                metadata = self._get_rank_metadata(rank_entries, r)
                sc = normalized.get(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0})
                pr = parts.get(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0})
                rc = raw.get(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0})
                total = sc.get('GLOBAL', 0) + sc.get('TP', 0) + sc.get('PP', 0) + sc.get('DP', 0)
                gl_fmt = f"{sc.get('GLOBAL',0)}/{pct(rc.get('GLOBAL',0), pr.get('GLOBAL',0)):.1f}%"
                tp_fmt = f"{sc.get('TP',0)}/{pct(rc.get('TP',0), pr.get('TP',0)):.1f}%"
                pp_fmt = f"{sc.get('PP',0)}/{pct(rc.get('PP',0), pr.get('PP',0)):.1f}%"
                dp_fmt = f"{sc.get('DP',0)}/{pct(rc.get('DP',0), pr.get('DP',0)):.1f}%"
                self.logger.info(f"    {r:<6} {metadata['node_ip']:<16} {metadata['hostname']:<16} {metadata['gpu_pci']:<16} {gl_fmt:<16} {tp_fmt:<16} {pp_fmt:<16} {dp_fmt:<16} {total:<10}")
            
            self.logger.info("  • Cumulative Slow Time (seconds):")
            self.logger.info(f"    {'Rank':<6} {'node_ip':<16} {'hostname':<16} {'gpu_pci':<16} {'GLOBAL':<12} {'TP':<12} {'PP':<12} {'DP':<12} {'Total':<12}")
            self.logger.info("    " + "-" * 132)
            for r in rank_list:
                metadata = self._get_rank_metadata(rank_entries, r)
                dur = durations.get(r, {'GLOBAL':0.0,'TP':0.0,'PP':0.0,'DP':0.0})
                total_d = float(dur.get('GLOBAL',0.0)) + float(dur.get('TP',0.0)) + float(dur.get('PP',0.0)) + float(dur.get('DP',0.0))
                self.logger.info(f"    {r:<6} {metadata['node_ip']:<16} {metadata['hostname']:<16} {metadata['gpu_pci']:<16} {dur.get('GLOBAL',0.0):<12.6f} {dur.get('TP',0.0):<12.6f} {dur.get('PP',0.0):<12.6f} {dur.get('DP',0.0):<12.6f} {total_d:<12.6f}")
            
            return parallel
            
        except Exception as e:
            self.logger.error(f"Error in parallel slow analysis: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _analyze_grouphash_slow(self, rank_entries: Dict[int, List[LogEntry]]) -> Optional[Dict]:
        """Analyze GroupHash-based slow detection."""
        try:
            # GroupHashSlowDetector is already imported at the top of the file
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("GROUP HASH BASED SLOW DETECTION ANALYSIS")
            self.logger.info("=" * 80)
            
            grouphash_detector = GroupHashSlowDetector(
                verbose=self.verbose,
                use_multiprocessing=True,
                rank_entries=rank_entries
            )
            
            if self.verbose:
                self.logger.info("Running GroupHash-based slow detection analysis...")
            
            operations_by_group = grouphash_detector.parse_all_logs(rank_entries)
            
            if not operations_by_group:
                self.logger.info("  • No operations with groupHash found in logs")
                return None
            
            group_performance = grouphash_detector.analyze_group_performance()
            
            if not group_performance:
                self.logger.info("  • GroupHash performance analysis failed")
                return None
            
            # Collect summary information
            total_groups = len(group_performance)
            total_slow_picks = sum(1 for gp in group_performance.values() if gp.is_outlier)
            total_operations = sum(len(ops) for ops in operations_by_group.values())
            total_ranks = len(set().union(*[gp.ranks for gp in group_performance.values()]))
            
            # Generate slow rank matrix
            matrix_data = grouphash_detector.generate_slow_rank_matrix()
            grouphash_results = {
                'total_groups': total_groups,
                'total_slow_picks': total_slow_picks,
                'total_operations': total_operations,
                'total_ranks': total_ranks,
                'matrix_data': matrix_data,
                'group_performance': {gh: {
                    'group_id': gp.group_id,
                    'op_count': gp.op_count,
                    'ranks': gp.ranks,
                    'slowest_rank': gp.slowest_rank,
                    'slowest_time': gp.slowest_time,
                    'is_outlier': gp.is_outlier
                } for gh, gp in group_performance.items()}
            }
            
            if matrix_data:
                self.logger.info("  • GroupHash Slow Rank Matrix (Rank vs Group, Normalized; non-participants shown as '-'):")
                all_groups = sorted(grouphash_detector.group_mapping.values())
                header = f"    {'Rank':<6} {'node_ip':<16} {'hostname':<16} {'gpu_pci':<16}"
                for group_id in all_groups:
                    header += f"{'G'+str(group_id):<6}"
                header += f"{'Total':<8}"
                self.logger.info(header)
                self.logger.info("    " + "-" * (6 + 16 * 3 + 6 * len(all_groups) + 8))
                
                for row in matrix_data:
                    rank = row[0]
                    values = row[1:]
                    total_slow = sum(v for v in values if isinstance(v, int))
                    metadata = self._get_rank_metadata(rank_entries, rank)
                    row_str = f"    {rank:<6} {metadata['node_ip']:<16} {metadata['hostname']:<16} {metadata['gpu_pci']:<16}"
                    for v in values:
                        cell = ('-' if v is None else str(v))
                        row_str += f"{cell:<6}"
                    row_str += f"{total_slow:<8}"
                    self.logger.info(row_str)
                
                self.logger.info("    Note: Per-group min computed over participants only; '-' means non-participant")
            
            # Verbose output (before summary)
            if self.verbose:
                self.logger.info("\n" + "-" * 80)
                self.logger.info("DETAILED GROUP HASH ANALYSIS:")
                self.logger.info("-" * 80)
                grouphash_detector.print_parallel_style_summary()
            
            # Output separator and summary (always at the end)
            self.logger.info("=" * 80)
            self.logger.info("GROUP HASH SUMMARY:")
            self.logger.info(f"  • Groups Analyzed: {total_groups}")
            self.logger.info(f"  • Total Operations: {total_operations}")
            self.logger.info(f"  • Total Ranks: {total_ranks}")
            self.logger.info(f"  • Slow Picks: {total_slow_picks}")
            
            # Print top slow ranks
            rank_slow_totals = {}
            for rank, group_counts in grouphash_detector.rank_slow_counts.items():
                rank_slow_totals[rank] = sum(group_counts.values())
            
            if rank_slow_totals:
                sorted_slow_ranks = sorted(rank_slow_totals.items(), key=lambda x: x[1], reverse=True)
                top_slow_ranks = [r for r, count in sorted_slow_ranks if count > 0][:3]
                
                if top_slow_ranks:
                    self.logger.info(f"  • Top Slow Ranks:")
                    for rank in top_slow_ranks:
                        metadata = self._get_rank_metadata(rank_entries, rank)
                        self.logger.info(f"    Rank {rank}: Hostname={metadata['hostname']}, IP={metadata['node_ip']}, PCI={metadata['gpu_pci']}")
            
            self.logger.info("=" * 80)
            return grouphash_results
            
        except Exception as e:
            self.logger.error(f"  • GroupHash analysis failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None


def load_rank_entries(analyzer: DistributedLogAnalyzer, log_path: Optional[str], verbose: bool,max_save_count_groups: int = 2):
    # if not analyzer.log_path:
    #     print("No log path specified!")
    #     return {}
    reader = LogReader(
        log_path=str(log_path),
        max_save_count_groups=max_save_count_groups,
        logger=analyzer.logger,
    )
    log_files = reader.discover_log_files()
    if not log_files:
        print("No log files found or parsing failed!")
        return {}
    if verbose:
        print(f"[DEBUG] Discovered {len(log_files)} log files under {log_path}")
    rank_entries = reader.parse_log_files()
    if verbose:
        total_lines = sum(len(entries) for entries in rank_entries.values())
        print(f"[DEBUG] Parsed {len(rank_entries)} ranks, total {total_lines} lines")
    return rank_entries


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Distributed Training Log Hang Detection Analyzer')
    parser.add_argument('log_path', help='Path to log file or directory containing log files to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Load logs with LogReader, then analyze
    reader = LogReader(log_path=args.log_path)
    reader.discover_log_files()
    rank_entries = reader.parse_log_files()
    
    analyzer = DistributedLogAnalyzer(args.log_path, args.verbose)
    analyzer.run_hang(rank_entries)


if __name__ == '__main__':
    main() 