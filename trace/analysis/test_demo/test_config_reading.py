#!/usr/bin/env python3
"""
Config reader sanity check for TP/PP/world_size and DP calculation.
Note: Unified CLI handles config validation automatically; keep for quick manual checks.
"""

import os
import yaml


def main():
    print("Config reading and DP calculation check")
    print("=" * 50)

    config_file = 'analysis/config_invalid.yaml'  # Intentional invalid-case sample
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found!")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        tp_size = config.get('TP', 2)
        pp_size = config.get('PP', 2)
        n_workers = config.get('world_size', 8)

        print(f"Configuration from {config_file}:")
        print(f"  • TP: {tp_size}")
        print(f"  • PP: {pp_size}")
        print(f"  • world_size: {n_workers}")

        if tp_size * pp_size > n_workers:
            print("\nInvalid configuration - TP * PP > world_size")
            return

        dp_size = n_workers // (tp_size * pp_size)
        print("\nDP Calculation:")
        print(f"  • DP = {n_workers} // ({tp_size} * {pp_size}) = {dp_size}")
        if dp_size == 0:
            print("Invalid configuration - DP_SIZE cannot be 0")
            return

        print("\nValid configuration")
        print(f"  • TP_SIZE: {tp_size}")
        print(f"  • PP_SIZE: {pp_size}")
        print(f"  • DP_SIZE: {dp_size}")
        print(f"  • N_WORKERS: {n_workers}")

        print("\nRank Distribution Example (first few ranks):")
        for i in range(min(8, n_workers)):
            pp_stage = i // (n_workers // pp_size)
            tp_pos = i % tp_size
            dp_pos = (i // tp_size) % dp_size
            print(f"  Rank {i}: TP={tp_pos}, PP={pp_stage}, DP={dp_pos}")
    except Exception as e:
        print(f"Error reading config file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
