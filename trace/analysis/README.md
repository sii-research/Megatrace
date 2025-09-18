## Distributed Training Log Analyzer 

Tools in this folder help diagnose and analyze distributed training logs: hang detection, slow-rank analysis (single-rank and parallel TP/PP/DP), 3D parallel configuration, and optional time-pattern analysis.
To ensure broad applicability of the tool, we have minimized its dependency on model configuration inputs. The corresponding analysis can be performed using only parallelization information.

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

### Slowdown simulation

```bash
python simulator/gpu_interferer.py --device 0 --duration 120 --workers 4 --matrix-size 8192

```


### Roadmap
- [ ] Identify loop information and directly obtain iteration positions from the NCCL side 
- [ ] Locate abnormal nodes using LSTM-based time series analysis
- [x] Locate computation slowness by analyzing synchronization call time differences
- [x] Detect hangs through missing call analysis


### Notes
- Paths in this README are relative to analysis/ (no "analysis/" prefix).
- Logs folder: ./logs
- Config file: ./config.yaml

### example 

- hang detect
```bash
$ python test_analyzer.py --log-path logs/log_hang/  --test-type hang

Unique opCounts (appearing only once):
  opCount 155: Rank 3

Minimum unique opCount: 155 (Rank 3)

Details for Rank 3 (minimum unique opCount):
  Stream 0x557c2d357250: AllGather (opCount 155)

```
Rank3 is a hang rank.


- Slowdown detect 
```bash
$ python test_analyzer.py --log-path logs/log_slow/  --test-type slow

PARALLEL SLOW NODE ANALYSIS:
  • Configuration: TP=2, PP=2, DP=1
  • Groups Analyzed: 44428
  • Slow Picks: 44428
  • Normalized Slow Counts / Rate (relative to min per type; rate = raw/participations):
    Rank   GLOBAL           TP               PP               DP               Total
    ----------------------------------------------------------------------------------------------
    0      0/0.0%           3821/69.0%       0/0.0%           0/0.0%           3821
    1      0/0.0%           0/31.0%          0/0.0%           0/0.0%           0
    2      0/0.0%           0/40.3%          0/0.0%           0/0.0%           0
    3      0/0.0%           1863/59.7%       0/0.0%           0/0.0%           1863
    4      0/0.0%           6889/85.8%       0/0.0%           0/0.0%           6889
    5      0/0.0%           0/14.2%          0/0.0%           0/0.0%           0
    6      0/0.0%           7149/73.6%       0/0.0%           0/0.0%           7149
    7      0/0.0%           0/26.4%          0/0.0%           0/0.0%           0
  • Cumulative Slow Time (seconds):
    Rank   GLOBAL       TP           PP           DP           Total
    ------------------------------------------------------------------------------
    0      0.000000     4.112090     0.000000     0.000000     4.112090
    1      0.000000     1.270806     0.000000     0.000000     1.270806
    2      0.000000     5.517180     0.000000     0.000000     5.517180
    3      0.000000     1.749636     0.000000     0.000000     1.749636
    4      0.000000     15.956919    0.000000     0.000000     15.956919
    5      0.000000     1.045251     0.000000     0.000000     1.045251
    6      0.000000     8.065365     0.000000     0.000000     8.065365
    7      0.000000     1.102379     0.000000     0.000000     1.102379


```
Rank4 is a slow rank.

