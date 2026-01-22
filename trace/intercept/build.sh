export CPATH=/usr/local/cuda/include:$CPATH
g++ -shared -o nccl_intercept.so nccl_intercept.cc ring_log.cc -ldl -fPIC -lpthread
