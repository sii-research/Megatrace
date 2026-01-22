export NCCL_MEGATRACE_ENABLE=1
export NCCL_MEGATRACE_LOG_PATH=./output
export LD_PRELOAD=../intercept/nccl_intercept.so
torchrun \
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr 127.0.0.1 \
--master_port 22234 test.py