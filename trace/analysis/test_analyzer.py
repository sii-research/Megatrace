#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Training Log Analyzer - Unified CLI
- Input parameter management
- Output rendering (concise by default; detailed with --verbose)
- Orchestration for hang detection, slow detection, and parallel analysis
"""

import argparse
import sys
import os
import yaml

# Ensure module import from current directory (so `from analysis import ...` uses analysis.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import DistributedLogAnalyzer  # noqa: E402
from group_hash_slow_detector import GroupHashSlowDetector  # noqa: E402


def run_hang(analyzer: DistributedLogAnalyzer, verbose: bool) -> bool:
    analyzer.discover_log_files()
    rank_entries = analyzer.parse_log_files()
    if not rank_entries:
        print("No log data found or parsing failed!")
        return False
    last_group = analyzer.get_last_save_count_group(rank_entries)
    streams = analyzer.group_by_stream(last_group)
    last_ops = analyzer.find_last_operation_in_streams(streams)
    hangs = analyzer.detect_hangs(last_ops, streams)

    print("\n" + "=" * 60)
    print("Distributed Training Log Analyzer - Hang Detection")
    print("=" * 60)
    print(f"Testing log path: {analyzer.log_path}/")

    print("\n" + "=" * 60)
    print("Analysis Summary:")
    print("=" * 60)
    unique_ranks = len(last_ops)
    total_streams = sum(len(v) for v in streams.values())
    print(f"Total Ranks: {unique_ranks}")
    print(f"Total Streams: {total_streams}")

    print("\nLast Operations by Rank and Stream:")
    for rank in sorted(last_ops.keys()):
        print(f"Rank {rank}:")
        for stream in sorted(last_ops[rank].keys()):
            op = last_ops[rank][stream]
            print(f"  Stream {stream}: {op.function} (opCount {op.op_count})")

    print("\n" + "=" * 60)
    print("Summary Analysis - Synchronization Comparison:")
    print("=" * 60)
    opcount_map = {}
    for rank, sdict in last_ops.items():
        for stream, op in sdict.items():
            opcount_map.setdefault(op.op_count, []).append((rank, op.function, stream))

    duplicate_items = [(oc, vals) for oc, vals in sorted(opcount_map.items()) if len(vals) > 1]
    if duplicate_items:
        print("\nDuplicate opCounts (appearing in multiple ranks):")
        for oc, vals in duplicate_items:
            ranks = sorted(r for r, _, _ in vals)
            print(f"  opCount {oc}: Ranks {ranks}")
            op_type_ranks = {}
            for r, func, _ in vals:
                op_type_ranks.setdefault(func, []).append(r)
            for func in sorted(op_type_ranks.keys()):
                print(f"    {func}: Ranks {sorted(set(op_type_ranks[func]))}")

    unique_items = [(oc, vals[0]) for oc, vals in sorted(opcount_map.items()) if len(vals) == 1]
    if unique_items:
        print("\nUnique opCounts (appearing only once):")
        for oc, (r, _, _) in unique_items:
            print(f"  opCount {oc}: Rank {r}")
        min_oc, (min_rank, _, _) = min(unique_items, key=lambda x: x[0])
        print(f"\nMinimum unique opCount: {min_oc} (Rank {min_rank})")
        for stream, op in streams[min_rank].items():
            lop = last_ops[min_rank].get(stream)
            if lop and lop.op_count == min_oc:
                print("\nDetails for Rank {0} (minimum unique opCount):".format(min_rank))
                print(f"  Stream {stream}: {lop.function} (opCount {lop.op_count})")
                break

    print("\n" + "=" * 60)
    print("Hang Detection Completed!")
    print("=" * 60)
    return True


def run_slow(analyzer: DistributedLogAnalyzer, logs_path: str, verbose: bool, config_path: str) -> bool:
    analyzer.discover_log_files()
    rank_entries = analyzer.parse_log_files()
    if not rank_entries:
        print("No log data found or parsing failed!")
        return False

    # Only analyze the last save_count group per rank; cross-group patterns are not meaningful
    last_group_entries = analyzer.get_last_save_count_group(rank_entries)

    # Multithreaded per-rank slow analysis
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def compute_rank_metrics(rank, entries):
        import numpy as np  # local import to avoid global overhead
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

    # Fixed to 8 threads as per design
    max_workers = 8
    if verbose:
        print(f"\nRunning multithreaded slow analysis with {max_workers} workers (per-rank, last save_count group)...")
    rank_metrics = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_rank_metrics, rank, entries) for rank, entries in last_group_entries.items()]
        for fut in as_completed(futures):
            rank, metrics = fut.result()
            if metrics is not None:
                rank_metrics[rank] = metrics
    
    if not rank_metrics:
        print("  • No rank metrics computed")
    else:
        # Robust, threshold-based anomaly detection
        import numpy as np
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

        print("\nSIMPLE SLOW RANK ANALYSIS (multithreaded):")
        print(f"  • Total Ranks Analyzed: {len(rank_metrics)}")
        if slow_ranks:
            print(f"  • Slow Ranks Detected: {sorted(slow_ranks)}")
            for r in sorted(slow_ranks):
                rm = rank_metrics[r]
                cv_val = (cv_arr[ranks.index(r)])
                print(f"    Rank {r}:")
                print(f"      • Score: {rm['performance_score']:.3f}")
                print(f"      • Calls/sec: {rm['calls_per_second']:.2f} (total_calls={rm['total_calls']}, duration={rm['total_duration']:.2f}s)")
                print(f"      • Interval mean/std/p95/max: {rm['mean_interval']:.4f}s / {rm['std_interval']:.4f}s / {rm['p95_interval']:.4f}s / {rm['max_interval']:.4f}s")
                print(f"      • CV: {cv_val:.3f}")
        else:
            print("  • No obvious anomalies.")

    cfg_file = config_path
    if not os.path.exists(cfg_file):
        if verbose:
            print("Config file not found; skipping parallel slow analysis")
        return True
    cfg = yaml.safe_load(open(cfg_file, 'r', encoding='utf-8'))
    tp = cfg.get('TP', 2)
    pp = cfg.get('PP', 2)
    ws = cfg.get('world_size', 8)
    if tp * pp > ws:
        print("Invalid config (TP*PP > world_size); skip parallel slow analysis")
        return True
    parallel = analyzer.analyze_parallel_slow_nodes(rank_entries, tp, pp, ws)
    if 'error' in parallel:
        print(f"Parallel slow analysis error: {parallel['error']}")
        return False
    print("\nPARALLEL SLOW NODE ANALYSIS:")
    print(f"  • Configuration: TP={tp}, PP={pp}, DP={ws // (tp*pp)}")
    print(f"  • Groups Analyzed: {parallel['total_groups_analyzed']}")
    if 'total_slow_picks' in parallel:
        print(f"  • Slow Picks: {parallel['total_slow_picks']}")

    # Detailed per-rank table (rows=ranks, columns=TP/PP/DP/Total)
    normalized = parallel.get('normalized_scores', {}) or {}
    raw = parallel.get('raw_scores', {}) or {}
    durations = parallel.get('slow_durations', {}) or {}
    parts = parallel.get('participations', {}) or {}
    # base rank list from parsed entries to ensure full table even when no outliers
    rank_list = sorted(rank_entries.keys())
    # Prefer normalized; fallback to raw when normalized is empty
    # Always show normalized counts and cumulative durations
    print("  • Normalized Slow Counts / Rate (relative to min per type; rate = raw/participations):")
    print(f"    {'Rank':<6} {'GLOBAL':<16} {'TP':<16} {'PP':<16} {'DP':<16} {'Total':<10}")
    print("    " + "-" * 94)
    for r in rank_list:
        sc = normalized.get(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0})
        pr = parts.get(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0})
        rc = raw.get(r, {'GLOBAL':0,'TP':0,'PP':0,'DP':0})
        total = sc.get('GLOBAL', 0) + sc.get('TP', 0) + sc.get('PP', 0) + sc.get('DP', 0)
        def pct(raw_val, part_val):
            return 0.0 if part_val == 0 else (100.0 * float(raw_val) / float(part_val))
        gl_fmt = f"{sc.get('GLOBAL',0)}/{pct(rc.get('GLOBAL',0), pr.get('GLOBAL',0)):.1f}%"
        tp_fmt = f"{sc.get('TP',0)}/{pct(rc.get('TP',0), pr.get('TP',0)):.1f}%"
        pp_fmt = f"{sc.get('PP',0)}/{pct(rc.get('PP',0), pr.get('PP',0)):.1f}%"
        dp_fmt = f"{sc.get('DP',0)}/{pct(rc.get('DP',0), pr.get('DP',0)):.1f}%"
        print(f"    {r:<6} {gl_fmt:<16} {tp_fmt:<16} {pp_fmt:<16} {dp_fmt:<16} {total:<10}")

    print("  • Cumulative Slow Time (seconds):")
    print(f"    {'Rank':<6} {'GLOBAL':<12} {'TP':<12} {'PP':<12} {'DP':<12} {'Total':<12}")
    print("    " + "-" * 78)
    for r in rank_list:
        dur = durations.get(r, {'GLOBAL':0.0,'TP':0.0,'PP':0.0,'DP':0.0})
        total_d = float(dur.get('GLOBAL',0.0)) + float(dur.get('TP',0.0)) + float(dur.get('PP',0.0)) + float(dur.get('DP',0.0))
        print(f"    {r:<6} {dur.get('GLOBAL',0.0):<12.6f} {dur.get('TP',0.0):<12.6f} {dur.get('PP',0.0):<12.6f} {dur.get('DP',0.0):<12.6f} {total_d:<12.6f}")
    
    # Add GroupHash-based slow detection analysis
    print("\n" + "=" * 80)
    print("GROUP HASH BASED SLOW DETECTION ANALYSIS")
    print("=" * 80)
    
    try:
        # Initialize GroupHash detector
        grouphash_detector = GroupHashSlowDetector(
            logs_path=logs_path,
            verbose=verbose,
            use_multiprocessing=True
        )
        
        if verbose:
            print("Running GroupHash-based slow detection analysis...")
        
        # Parse logs and analyze
        operations_by_group = grouphash_detector.parse_all_logs()
        
        if not operations_by_group:
            print("  • No operations with groupHash found in logs")
        else:
            # Analyze performance
            group_performance = grouphash_detector.analyze_group_performance()
            
            if not group_performance:
                print("  • GroupHash performance analysis failed")
            else:
                # Print concise summary
                total_groups = len(group_performance)
                total_slow_picks = sum(1 for gp in group_performance.values() if gp.is_outlier)
                total_operations = sum(len(ops) for ops in operations_by_group.values())
                total_ranks = len(set().union(*[gp.ranks for gp in group_performance.values()]))
                
                print(f"  • Groups Analyzed: {total_groups}")
                print(f"  • Total Operations: {total_operations}")
                print(f"  • Total Ranks: {total_ranks}")
                print(f"  • Slow Picks: {total_slow_picks}")
                
                # Generate slow rank matrix
                matrix_data = grouphash_detector.generate_slow_rank_matrix()
                if matrix_data:
                    print("  • GroupHash Slow Rank Matrix (Rank vs Group, Normalized; non-participants shown as '-'):")
                    
                    # Print header
                    all_groups = sorted(grouphash_detector.group_mapping.values())
                    header = f"    {'Rank':<6}"
                    for group_id in all_groups:
                        header += f"{'G'+str(group_id):<6}"
                    header += f"{'Total':<8}"
                    print(header)
                    print("    " + "-" * (6 + 6 * len(all_groups) + 8))
                    
                    # Print matrix rows (None -> '-') and totals ignoring None
                    for row in matrix_data:
                        rank = row[0]
                        values = row[1:]
                        total_slow = sum(v for v in values if isinstance(v, int))
                        row_str = f"    {rank:<6}"
                        for v in values:
                            cell = ('-' if v is None else str(v))
                            row_str += f"{cell:<6}"
                        row_str += f"{total_slow:<8}"
                        print(row_str)
                    
                    # Print normalization info
                    print("    Note: Per-group min computed over participants only; '-' means non-participant")
                
                # Print top slow ranks
                rank_slow_totals = {}
                for rank, group_counts in grouphash_detector.rank_slow_counts.items():
                    rank_slow_totals[rank] = sum(group_counts.values())
                
                if rank_slow_totals:
                    sorted_slow_ranks = sorted(rank_slow_totals.items(), key=lambda x: x[1], reverse=True)
                    top_slow_ranks = [r for r, count in sorted_slow_ranks if count > 0][:3]  # Top 3
                    
                    if top_slow_ranks:
                        print(f"  • Top Slow Ranks: {top_slow_ranks}")
                        for rank in top_slow_ranks:
                            total_slow = rank_slow_totals[rank]
                            print(f"    Rank {rank}: {total_slow} slow operations")
                
                # Print detailed analysis if verbose
                if verbose:
                    print("\n" + "-" * 60)
                    print("DETAILED GROUP HASH ANALYSIS:")
                    print("-" * 60)
                    grouphash_detector.print_parallel_style_summary()
    
    except Exception as e:
        print(f"  • GroupHash analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    print("=" * 80)
    return True


def run_parallel_config(analyzer: DistributedLogAnalyzer, verbose: bool, config_path: str) -> bool:
    cfg_file = config_path
    if not os.path.exists(cfg_file):
        print(f"Config file '{cfg_file}' not found; skipping parallel config analysis")
        return True
    cfg = yaml.safe_load(open(cfg_file, 'r', encoding='utf-8'))
    tp = cfg.get('TP', 2)
    pp = cfg.get('PP', 2)
    ws = cfg.get('world_size', 8)
    if tp * pp > ws:
        print("Invalid configuration (TP * PP > world_size); skipping parallel config analysis")
        return True
    ranks_info = analyzer.analyze_3d_parallel_ranks(tp, pp, ws)
    groups = analyzer.analysis_results.get('3d_parallel_groups', {})
    analyzer.print_3d_parallel_summary(ranks_info, groups.get('tp_groups', []), groups.get('pp_groups', []), groups.get('dp_groups', []))
    return True


def main():
    parser = argparse.ArgumentParser(description='Distributed Training Log Analyzer - Unified CLI')
    parser.add_argument('--log-path', required=True, help='Path to directory containing log files')
    parser.add_argument('--test-type', choices=['hang', 'slow', 'parallel', 'all'], default='all', help='Type of analysis to perform')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output with detailed progress')
    parser.add_argument('--config-path', default='config.yaml', help='Path to parallel analysis config (default: config.yaml)')
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log path '{args.log_path}' does not exist!")
        return
    
    analyzer = DistributedLogAnalyzer(args.log_path, verbose=args.verbose)
    ok = True
    if args.test_type == 'hang':
        ok = run_hang(analyzer, args.verbose)
    elif args.test_type == 'slow':
        ok = run_slow(analyzer, args.log_path, args.verbose, args.config_path)
    elif args.test_type == 'parallel':
        ok = run_parallel_config(analyzer, args.verbose, args.config_path)
    else:
        ok = run_hang(analyzer, args.verbose)
        ok = run_slow(analyzer, args.log_path, args.verbose, args.config_path) and ok
        ok = run_parallel_config(analyzer, args.verbose, args.config_path) and ok
    print("\nAnalysis completed successfully!" if ok else "\nAnalysis failed!")


if __name__ == '__main__':
    main()


