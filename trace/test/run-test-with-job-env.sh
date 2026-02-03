#!/usr/bin/env bash
# Run test.py with "case 1" env: TRAIN_JOB_ID, POD (RUNNING_ROUND), MY_POD_IP;
# logs will have prefix [TRAIN_JOB_ID] [RUNNING_ROUND] [node_ip] [hostname] [save_count N] ...
# Run from trace/test/ (same as run-test.sh).

export NCCL_MEGATRACE_ENABLE=1
export NCCL_MEGATRACE_LOG_PATH=./output

# Extra log prefix fields: TRAIN_JOB_ID (env) + RUNNING_ROUND (computed from POD)
export MEGATRACE_LOG_EXTRA_FIELDS=TRAIN_JOB_ID,RUNNING_ROUND

# Job/pod-like env (for log prefix and RUNNING_ROUND computation)
export TRAIN_JOB_ID="job-9f34aa02-f3f9-4fcf-ab45-56f8e61baaae"
export POD="job-9f34aa02-f3f9-4fcf-ab45-56f8e61baaae-worker-0-0"
export MY_POD_IP="${MY_POD_IP:-10.1.3.201}"

export LD_PRELOAD="${LD_PRELOAD:-../intercept/nccl_intercept.so}"

torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 22234 \
  test.py
