
# Simulated deceleration
## Basic interference: on GPU 0, duration 120 seconds, 4 parallel streams, matrix size 8192
python gpu_interferer.py --device 0 --duration 120 --workers 4 --matrix-size 8192

## Add 6GB VRAM pressure, use float16, 10 ms sleep between iterations
python gpu_interferer.py --device 0 --duration 120 --workers 4 --matrix-size 8192 --dtype float16 --mem-gb 6 --sleep-ms 10