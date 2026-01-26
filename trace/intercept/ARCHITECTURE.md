# 集合通信拦截架构文档

## 架构概述

本架构采用**共享基础代码 + 厂商特定实现分离**的设计，支持 NVIDIA (NCCL) 和 AMD (RCCL) 两种 GPU 厂商的集合通信拦截。

## 文件结构

```
trace/intercept/
├── intercept_common.h          # 共享的类型定义和接口
├── intercept_common.cc         # 共享的基础功能实现
├── nccl_intercept.cc           # NVIDIA NCCL 实现（条件编译）
├── rccl_intercept.cc           # AMD RCCL 实现（条件编译）
├── ring_log.cc                 # 日志系统（共享）
├── ring_log.h                  # 日志头文件
├── gpu_config.h                # GPU配置抽象层
├── intercept.h                 # 拦截函数类型定义（支持多厂商）
└── build.sh                    # 构建脚本（根据GPU厂商选择源文件）
```

## 设计原则

### 1. 代码分离
- **共享代码**：提取到 `intercept_common.h/cc`
  - 哈希函数实现
  - 日志线程初始化
  - 符号解析通用逻辑
  - Stream操作计数

- **厂商特定代码**：分别实现
  - `nccl_intercept.cc`：NCCL 特定实现
  - `rccl_intercept.cc`：RCCL 特定实现

### 2. 条件编译
- 使用 `#ifdef MEGATRACE_GPU_NVIDIA` 和 `#ifdef MEGATRACE_GPU_AMD` 确保每个文件只编译对应厂商的代码
- 避免类型冲突和未使用的代码

### 3. 类型抽象
- 在 `intercept.h` 中根据 GPU 厂商包含不同的头文件
- 使用类型别名处理 API 差异
- 保持接口一致性

## 关键组件

### intercept_common.h/cc
提供厂商无关的共享功能：
- `hashUniqueId()`: 通用哈希函数
- `init_log_writer_thread()`: 日志线程初始化
- `resolve_symbol_common()`: 通用符号解析模板
- `next_opcount_for_stream()`: Stream操作计数

### nccl_intercept.cc
NCCL 特定实现：
- Comm 映射管理（使用 NCCL 类型）
- NCCL 函数拦截
- NCCL 库搜索列表
- 条件编译：`#ifdef MEGATRACE_GPU_NVIDIA`

### rccl_intercept.cc
RCCL 特定实现：
- Comm 映射管理（使用兼容类型）
- RCCL 函数拦截（支持 nccl* 和 rccl* 命名）
- RCCL 库搜索列表
- 条件编译：`#ifdef MEGATRACE_GPU_AMD`

### intercept.h
根据 GPU 厂商：
- 包含对应的头文件（`nccl.h` 或 `rccl.h`）
- 定义类型别名（如果需要）
- 提供统一的函数指针类型

## 构建流程

### NVIDIA 版本
```bash
./build.sh --nvidia
# 或
GPU_VENDOR=nvidia ./build.sh
```

编译文件：
- `nccl_intercept.cc`
- `intercept_common.cc`
- `ring_log.cc`

输出：`nccl_intercept.so`

### AMD 版本
```bash
./build.sh --amd
# 或
GPU_VENDOR=amd ./build.sh
```

编译文件：
- `rccl_intercept.cc`
- `intercept_common.cc`
- `ring_log.cc`

输出：`rccl_intercept.so`

## API 差异处理

### 函数命名
- **NCCL**: 使用 `nccl*` 函数名
- **RCCL**: 优先尝试 `nccl*`（兼容性），失败后尝试 `rccl*`

### 类型映射
在 `intercept.h` 中处理：
```cpp
#ifdef MEGATRACE_GPU_AMD
    #include <rccl.h>
    // 类型别名（如果需要）
    typedef rcclComm_t ncclComm_t;
    typedef rcclUniqueId ncclUniqueId;
#endif
```

### 库搜索
每个实现文件维护自己的库搜索列表：
- NCCL: `libnccl.so`, `libtorch_cuda.so` 等
- RCCL: `librccl.so`, `libtorch_hip.so` 等

## 代码复用

### 共享逻辑
以下逻辑在两个实现中共享（通过 `intercept_common`）：
- Comm ID 到 Comm 的映射逻辑
- UniqueId 哈希计算
- GroupHash 计算
- 日志线程管理
- Stream 操作计数

### 厂商特定逻辑
以下逻辑在每个实现中独立：
- Comm 映射数据结构（使用各自的类型）
- 符号解析的库搜索列表
- 函数拦截的具体实现（虽然逻辑相似，但函数名可能不同）

## 扩展性

### 添加新厂商
1. 创建新的实现文件（如 `intel_intercept.cc`）
2. 在 `intercept.h` 中添加对应的头文件包含和类型映射
3. 在 `build.sh` 中添加新的编译选项
4. 实现厂商特定的拦截函数

### 添加新功能
1. 如果是共享功能，添加到 `intercept_common.h/cc`
2. 如果是厂商特定功能，在对应的实现文件中添加

## 优势

1. **代码清晰**：每个厂商的实现独立，易于理解和维护
2. **减少重复**：共享逻辑提取到公共文件
3. **易于扩展**：添加新厂商只需创建新文件
4. **编译时优化**：只编译需要的代码，减小二进制体积
5. **类型安全**：通过条件编译避免类型冲突

## 注意事项

1. **条件编译**：确保所有厂商特定代码都包裹在对应的 `#ifdef` 中
2. **类型一致性**：在共享代码中使用抽象类型，在实现中映射到具体类型
3. **符号解析**：不同厂商的库名和函数名可能不同，需要分别处理
4. **API 兼容性**：RCCL 可能保持 NCCL API 兼容，也可能使用不同的命名

## 测试建议

1. **编译测试**：分别测试 NVIDIA 和 AMD 版本的编译
2. **功能测试**：在实际环境中测试拦截功能
3. **符号解析测试**：验证能正确找到并调用原始函数
4. **日志测试**：验证日志记录功能正常

---

**最后更新**: 2024-01-XX

