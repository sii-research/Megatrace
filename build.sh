g++ -shared -o nccl_intercept.so nccl_intercept_main.cpp ring_log.cc -ldl -fPIC -lpthread
