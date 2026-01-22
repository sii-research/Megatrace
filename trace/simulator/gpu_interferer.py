#!/usr/bin/env python3
"""
GPU Interferer (PyTorch-based)

Purpose: Run heavy GPU workloads on a specified device to intentionally
interfere with/slow down other jobs on that GPU (for testing).

Usage (from analysis/):
  python gpu_interferer.py --device 0 --duration 120 --workers 4 --matrix-size 8192 --mem-gb 6

Notes:
- This script is intended for local testing in controlled environments.
- It will consume GPU compute and memory on the chosen device.
"""

import argparse
import time
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heavy GPU workload on a specified device")
    parser.add_argument("--device", type=int, required=True, help="CUDA device index to use")
    parser.add_argument("--duration", type=float, default=60.0, help="Run time in seconds (default: 60)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel CUDA streams (default: 4)")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Square matrix size for matmul (default: 8192)")
    parser.add_argument("--dtype", choices=["float16", "float32", "float64"], default="float32", help="Tensor dtype")
    parser.add_argument("--mem-gb", type=float, default=0.0, help="Allocate ~GB of device memory to increase pressure")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between batches per worker (ms)")
    return parser.parse_args()


def get_dtype(torch_module, name: str):
    if name == "float16":
        return torch_module.float16
    if name == "float64":
        return torch_module.float64
    return torch_module.float32


def allocate_pressure_memory(torch_module, device: int, mem_gb: float):
    if mem_gb <= 0:
        return None
    bytes_total = int(mem_gb * (1024 ** 3))
    elem_size = torch_module.tensor([], dtype=torch_module.float32).element_size()
    num_elems = bytes_total // elem_size
    if num_elems <= 0:
        return None
    try:
        return torch_module.empty(num_elems, dtype=torch_module.float32, device=device)
    except Exception as e:
        print(f"[warn] Memory pressure allocation failed ({e}); continuing without it")
        return None


def worker_loop(torch_module, device: int, matrix_size: int, dtype, end_time: float, sleep_ms: int, stream):
    torch_module.cuda.set_device(device)
    # Create persistent operands once per worker
    a = torch_module.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    b = torch_module.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    # Warmup
    with torch_module.no_grad():
        with torch_module.cuda.stream(stream):
            c = torch_module.matmul(a, b)
    torch_module.cuda.synchronize(device)

    iters = 0
    with torch_module.no_grad():
        while time.time() < end_time:
            with torch_module.cuda.stream(stream):
                # Chain a few compute ops to increase kernel load
                c = torch_module.matmul(a, b)
                c = torch_module.relu_(c)
                # Optional: another matmul to keep SMs busy
                c2 = torch_module.matmul(b, a)
                c2.add_(c)
            iters += 1
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
    # Ensure kernels finished
    torch_module.cuda.synchronize(device)
    return iters


def main():
    args = parse_args()
    try:
        import torch
    except Exception as e:
        raise SystemExit(f"PyTorch is required. Install via: python -m pip install torch\nError: {e}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Please ensure a CUDA-enabled PyTorch build and GPU are present.")

    device = args.device
    torch.cuda.set_device(device)

    # Select dtype
    dtype = get_dtype(torch, args.dtype)

    # Pre-allocate memory pressure if requested
    pressure = allocate_pressure_memory(torch, device, args.mem_gb)
    if pressure is not None:
        print(f"[info] Allocated ~{args.mem_gb} GB on device {device} for memory pressure")

    # Create CUDA streams
    streams: List[torch.cuda.Stream] = []
    for _ in range(max(1, args.workers)):
        # Lower (negative) priority = higher scheduling priority
        streams.append(torch.cuda.Stream(priority=-1))

    print(
        f"[info] Starting interference: device={device}, duration={args.duration}s, "
        f"workers={args.workers}, matrix_size={args.matrix_size}, dtype={args.dtype}, mem_gb={args.mem_gb}"
    )

    end_time = time.time() + float(args.duration)
    total_iters = 0
    for s in streams:
        total_iters += worker_loop(
            torch_module=torch,
            device=device,
            matrix_size=args.matrix_size,
            dtype=dtype,
            end_time=end_time,
            sleep_ms=args.sleep_ms,
            stream=s,
        )

    print(f"[info] Completed. Total worker iterations: {total_iters}")


if __name__ == "__main__":
    main()


