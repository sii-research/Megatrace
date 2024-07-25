# CudaEvent_intercept

## get project 

``` shell
git clone --recursive http://58.213.50.202:8081/moon/cudaevent_intercept.git
#or
git clone  http://58.213.50.202:8081/moon/cudaevent_intercept.git
cd cudaevent_intercept
git submodule update --init --recursive
```

## Getting started

```
vim ~/.bashrc 
export CPATH=/usr/local/cuda/include:$CPATH
g++ -shared -o watchdog.so cudaevent_intercept.cpp -ldl -fPIC
-x LD_PRELOAD=/path/watchdog.so \
```

