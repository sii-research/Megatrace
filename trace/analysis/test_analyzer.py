#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for hang and slow detection in analysis.py
Supports command-line arguments for flexible testing
"""

import argparse
import sys
import logging
import os
from pathlib import Path
from typing import Optional

# Add project root to path to ensure imports work
# Get the directory containing this script
script_dir = Path(__file__).parent.absolute()
# Get project root (parent of trace directory)
project_root = script_dir.parent.parent.absolute()

# Add both script directory and project root to path
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from analysis module in the same directory
from analysis import DistributedLogAnalyzer, load_rank_entries


def remove_datetime_from_logging():
    """Remove datetime from console logging output"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setFormatter(logging.Formatter('%(message)s'))


def test_hang_detection(log_path: str, verbose: bool = False, max_save_count_groups: int = 2):
    """
    Test hang detection functionality
    
    Args:
        log_path: Path to log file or directory
        verbose: Whether to show detailed output
        max_save_count_groups: Number of most recent save_count groups to load
    """
    try:
        # Initialize analyzer
        analyzer = DistributedLogAnalyzer(
            log_path=log_path,
            verbose=verbose,
            max_save_count_groups=max_save_count_groups
        )
        
        # Remove datetime from console output
        remove_datetime_from_logging()
        
        # Load rank entries
        rank_entries = load_rank_entries(analyzer, log_path, verbose, max_save_count_groups)
        
        if not rank_entries:
            return False
        
        # Run hang detection
        analyzer.run_hang(rank_entries, analyze_opcount=False)
        
        return True
            
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_slow_detection(log_path: str, verbose: bool = False, max_save_count_groups: int = 2, 
                       config_path: Optional[str] = None):
    """
    Test slow detection functionality

    """
    
    try:
        # Initialize analyzer
        analyzer = DistributedLogAnalyzer(
            log_path=log_path,
            verbose=verbose,
            max_save_count_groups=max_save_count_groups
        )
        
        # Remove datetime from console output
        remove_datetime_from_logging()
        
        # Load rank entries
        rank_entries = load_rank_entries(analyzer, log_path, verbose, max_save_count_groups)
        
        if not rank_entries:
            return False
        
        # Run slow detection
        # Note: config_path is optional and only used for parallel-aware detection
        analyzer.run_slow(
            rank_entries, 
            config_path=config_path,
            analyze_grouphash=True
        )
        
        return True
            
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Test hang and slow detection from analysis.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All analyses
  python test_analyzer.py --log-path logs --test-type all

  # Hang detection only
  python test_analyzer.py --log-path logs --test-type hang

  # Slow analysis (includes parallel-aware detection when a config is provided)
  python test_analyzer.py --log-path logs --test-type slow
        """
    )
    
    parser.add_argument(
        '--log-path',
        type=str,
        required=True,
        help='Path to log file or directory containing log files to analyze'
    )
    
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['all', 'hang', 'slow'],
        default='all',
        help='Type of test to run: all, hang, or slow (default: all)'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Optional path to parallel analysis config file (for parallel-aware slow detection)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output (enabled by default for test script)'
    )
    
    parser.add_argument(
        '--max-save-count-groups',
        type=int,
        default=2,
        help='Number of most recent save_count groups to load (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Default verbose to True for test script to show output
    if not args.verbose:
        args.verbose = True
    
    # Validate log path
    log_path = Path(args.log_path)
    if not log_path.exists():
        sys.exit(1)
    
    # Run tests based on test type
    if args.test_type in ['all', 'hang']:
        test_hang_detection(
            log_path=str(log_path),
            verbose=args.verbose,
            max_save_count_groups=args.max_save_count_groups
        )
    
    if args.test_type in ['all', 'slow']:
        test_slow_detection(
            log_path=str(log_path),
            verbose=args.verbose,
            max_save_count_groups=args.max_save_count_groups,
            config_path=args.config_path
        )


if __name__ == '__main__':
    main()
