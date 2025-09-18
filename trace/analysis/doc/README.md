## Distributed Training Log Analyzer (analysis/)

Tools in this folder help diagnose and analyze distributed training logs: hang detection, slow-rank analysis (single-rank and parallel TP/PP/DP), 3D parallel configuration, and optional time-pattern analysis.

### Requirements
- Python 3.7+
- numpy, pyyaml
- Optional: pandas (time pattern details), scipy (if extending stats)

Install dependencies (run inside analysis/):
```bash
python -m pip install -r requirements.txt
```
Notes:
- `requirements.txt` lists base deps by default. Uncomment optional lines to install extras.

### Expected Log Format
```
[save_count 1] [1755275536.335349923] [Rank 0] Fun AllReduce Data 1 stream 0x564f06ff8bd0 opCount 0
```

### Configuration (config.yaml)
```yaml
TP: 2
PP: 4
world_size: 8
# DP is auto-computed as world_size // (TP*PP)
```

### Unified CLI (recommended)
Run commands from this analysis/ directory.
```bash
# All analyses
python test_analyzer.py --log-path logs --test-type all --verbose

# Hang detection only
python test_analyzer.py --log-path logs --test-type hang

# Slow analysis (includes parallel-aware detection when a config is provided)
python test_analyzer.py --log-path logs --test-type slow --config-path config.yaml

# 3D parallel configuration analysis
python test_analyzer.py --log-path logs --test-type parallel --config-path config.yaml
```

### Output Highlights
- Hang detection: last operation per rank/stream and synchronization comparison.
- Slow analysis (single-rank):
  - Default output includes, for detected slow ranks: Score, Calls/sec (total_calls, duration), interval mean/std/p95/max, and CV.
  - Robust rule: report only when at least two of (low score, low calls/sec, high CV) significantly deviate from median-based thresholds.
- Parallel slow detection (TP/PP/DP groups):
  - IQR (Q3 + 1.5×IQR) plus Grubbs' test (groups of size 4–30).
  - Ranks x [TP, PP, DP, Total] table is printed by default (normalized when available).

### Standalone Utilities
- Parallel slow detector (for development/benchmark):
```bash
python parallel_slow_detector.py --logs-path logs --config-path config.yaml --verbose
```
- Time pattern analysis test:
```bash
python test/test_time_pattern.py --log-path logs --verbose
```

### Notes
- Paths in this README are relative to analysis/ (no "analysis/" prefix).
- Logs folder: ./logs
- Config file: ./config.yaml

