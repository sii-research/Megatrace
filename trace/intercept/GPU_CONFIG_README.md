# GPU 厂商配置说明

Megatrace 支持通过编译时宏来控制不同 GPU 厂商的编译逻辑，目前支持 NVIDIA (CUDA) 和 AMD (ROCm/HIP)。

## 快速开始

### 编译 NVIDIA 版本（默认）

```bash
# 方式1: 使用默认设置
./build.sh

# 方式2: 显式指定
./build.sh --nvidia

# 方式3: 使用环境变量
GPU_VENDOR=nvidia ./build.sh
```

### 编译 AMD 版本

```bash
# 方式1: 使用命令行参数
./build.sh --amd

# 方式2: 使用环境变量
GPU_VENDOR=amd ./build.sh
```

## 编译时宏定义

### NVIDIA (CUDA)

编译时定义 `MEGATRACE_GPU_NVIDIA`：

```bash
g++ -DMEGATRACE_GPU_NVIDIA -shared -o nccl_intercept.so ...
```

### AMD (ROCm/HIP)

编译时定义 `MEGATRACE_GPU_AMD`：

```bash
g++ -DMEGATRACE_GPU_AMD -shared -o rccl_intercept.so ...
```

## 代码中的使用

### 1. 包含 GPU 配置头文件

```c
#include "gpu_config.h"
```

### 2. 使用 GPU 抽象类型

```c
// 使用 gpu_stream_t 而不是 cudaStream_t 或 hipStream_t
gpu_stream_t stream;

// 使用 gpu_error_t 而不是 cudaError_t 或 hipError_t
gpu_error_t err = gpuGetDevice(&dev);

// 使用 GPU_SUCCESS 而不是 cudaSuccess 或 hipSuccess
if (err != GPU_SUCCESS) {
    // 错误处理
}
```

### 3. 条件编译

```c
#ifdef MEGATRACE_GPU_NVIDIA
    // NVIDIA 特定的代码
    #include <cublas_v2.h>
#elif defined(MEGATRACE_GPU_AMD)
    // AMD 特定的代码
    #include <hipblas.h>
#endif
```

### 4. 使用条件编译宏

```c
IF_NVIDIA(
    // 这段代码只在 NVIDIA 版本中编译
    cudaDeviceSynchronize();
)

IF_AMD(
    // 这段代码只在 AMD 版本中编译
    hipDeviceSynchronize();
)
```

## 类型映射

### NVIDIA (CUDA)

| 抽象类型 | CUDA 类型 |
|---------|-----------|
| `gpu_stream_t` | `cudaStream_t` |
| `gpu_error_t` | `cudaError_t` |
| `gpu_device_prop_t` | `cudaDeviceProp` |

### AMD (ROCm/HIP)

| 抽象类型 | HIP 类型 |
|---------|----------|
| `gpu_stream_t` | `hipStream_t` |
| `gpu_error_t` | `hipError_t` |
| `gpu_device_prop_t` | `hipDeviceProp_t` |

## 函数映射

所有 GPU 操作函数都通过宏映射到对应的厂商实现：

```c
// 这些宏会根据 GPU 厂商自动映射到正确的函数
gpuGetDevice(&dev);              // cudaGetDevice 或 hipGetDevice
gpuGetDeviceCount(&count);      // cudaGetDeviceCount 或 hipGetDeviceCount
gpuDeviceGetPCIBusId(...);      // cudaDeviceGetPCIBusId 或 hipDeviceGetPCIBusId
gpuStreamCreate(&stream);       // cudaStreamCreate 或 hipStreamCreate
```

## 库搜索

不同 GPU 厂商使用不同的库：

### NVIDIA
- `libtorch_cuda.so`
- `libnccl.so`

### AMD
- `libtorch_hip.so`
- `librccl.so`

库搜索列表在 `nccl_intercept.cc` 中通过条件编译自动选择。

## 环境变量

### NVIDIA 编译环境

```bash
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### AMD 编译环境

```bash
export ROCM_PATH=/opt/rocm
export CPATH=$ROCM_PATH/include:$CPATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

## 运行时信息

代码中可以通过以下宏获取当前 GPU 厂商信息：

```c
// 获取厂商名称字符串
MEGATRACE_GPU_VENDOR_NAME  // "NVIDIA" 或 "AMD"

// 获取 API 名称字符串
MEGATRACE_GPU_API_NAME     // "CUDA" 或 "HIP"
```

## 示例

### 获取 PCI 总线 ID（厂商无关）

```c
#include "gpu_config.h"

char pci_buf[32];
megatrace_get_pci_bus_id(pci_buf, sizeof(pci_buf));
// 自动使用正确的 API (cudaDeviceGetPCIBusId 或 hipDeviceGetPCIBusId)
```

### 条件编译示例

```c
#include "gpu_config.h"

void initialize_gpu() {
    int dev_count;
    gpuGetDeviceCount(&dev_count);
    
    #ifdef MEGATRACE_GPU_NVIDIA
        printf("Using NVIDIA CUDA with %d devices\n", dev_count);
    #elif defined(MEGATRACE_GPU_AMD)
        printf("Using AMD ROCm with %d devices\n", dev_count);
    #endif
}
```

## 故障排查

### 编译错误：找不到头文件

**NVIDIA:**
```bash
# 确保 CUDA 已安装
ls /usr/local/cuda/include/cuda_runtime.h

# 设置 CPATH
export CPATH=/usr/local/cuda/include:$CPATH
```

**AMD:**
```bash
# 确保 ROCm 已安装
ls $ROCM_PATH/include/hip/hip_runtime.h

# 设置 ROCM_PATH 和 CPATH
export ROCM_PATH=/opt/rocm
export CPATH=$ROCM_PATH/include:$CPATH
```

### 链接错误：找不到库

**NVIDIA:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**AMD:**
```bash
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### 运行时错误：符号未找到

确保使用正确的库版本：
- NVIDIA: 使用 `libnccl.so`
- AMD: 使用 `librccl.so`

## 扩展支持其他 GPU 厂商

要添加新的 GPU 厂商支持：

1. 在 `gpu_config.h` 中添加新的条件编译块
2. 定义相应的类型别名和函数映射
3. 更新 `build.sh` 添加新的编译选项
4. 更新 `nccl_intercept.cc` 中的库搜索列表

## 相关文件

- `gpu_config.h` - GPU 配置和抽象层定义
- `build.sh` - 构建脚本，支持 GPU 厂商选择
- `intercept.h` - 拦截函数类型定义
- `nccl_intercept.cc` - NCCL/RCCL 拦截实现
- `ring_log.h` / `ring_log.cc` - 日志系统

---

**最后更新**: 2024-01-XX

