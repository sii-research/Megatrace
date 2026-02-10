# 日志配置与解析说明

本文档说明 Megatrace 日志的**写端配置**（.so / ring_log）与**分析端解析**（log_reader）约定，便于统一格式与扩展。

---

## 一、日志写端配置（ring_log / .so）

### 1.1 环境变量总览

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `NCCL_MEGATRACE_ENABLE` | 是否开启 Megatrace | 1 |
| `NCCL_MEGATRACE_LOG_PATH` | 日志目录 | `./logs` |
| `NCCL_MEGATRACE_SENSTIME` | 落盘间隔（毫秒）：有数据时至少每隔该时间写一次文件 | 3000 |
| `MEGATRACE_LOG_EXTRA_FIELDS` | 可选，逗号分隔的“额外前缀字段”名（见下） | 无 |
| `MY_POD_IP` | 节点 IP（K8s 常用）；未设置时自动取本机主 IPv4 | - |
| `TRAIN_JOB_ID` | 训练任务 ID（可选，用于存储路径与 extra 前缀） | - |
| `POD` | Pod 名（可选，用于文件名与 RUNNING_ROUND 计算） | - |

### 1.2 日志行格式

每行格式为：

```
[可选前缀段...] [node_ip] [hostname] [save_count N] [消息体]
```

- **固定前缀**（分析依赖）：紧挨在 `[save_count N]` 前的两个 `[..]` 依次为 **node_ip**、**hostname**。
- **可选前缀**：由 `MEGATRACE_LOG_EXTRA_FIELDS` 控制，可有多段 `[value]`，顺序与配置一致，**必须**位于 node_ip、hostname 之前。
- **消息体**：由拦截层固定格式输出，包含 timestamp、Rank、PCI、Fun/Func、Data、stream、opCount、groupHash 等。

**约定**：写端在 node_ip、hostname 前可任意增加“可选前缀”段；分析端只认“最后两段为 node_ip、hostname”，其余前缀段忽略。

### 1.3 可选前缀字段（MEGATRACE_LOG_EXTRA_FIELDS）

- **格式**：逗号分隔的**字段名**，例如：`TRAIN_JOB_ID,RUNNING_ROUND`。
- **取值规则**：
  - **环境变量**：名字未注册为内置计算字段时，使用 `getenv(名字)`；未设置或空则不输出该段。
  - **内置计算字段**：当前支持 `RUNNING_ROUND`，由 resolver 根据 POD 名计算（POD 名最后一段，如 `worker-0-8` → `8`），不读环境变量 `RUNNING_ROUND`。
- **输出顺序**：与配置顺序一致，全部在 `[node_ip] [hostname]` 之前。

### 1.3.1 扩展内置计算字段（解耦设计）

计算逻辑已从写行循环中解耦，采用**注册表**方式。新增内置字段时无需修改 `ring_log.cc` 的循环。

- **实现位置**：`trace/intercept/extra_field_resolver.h`、`extra_field_resolver.cc`
- **扩展步骤（新增一个内置计算字段，如 `MY_ROUND`）**：
  1. 在 `extra_field_resolver.cc` 中实现 resolver 函数，例如：`std::string resolve_my_round(const ExtraFieldContext& ctx)`，从 `ctx.pod_name`、`ctx.save_iter` 等上下文计算字符串。
  2. 在 `init_builtin_extra_resolvers()` 中注册：`register_extra_field_resolver("MY_ROUND", resolve_my_round)`。
  3. 重新编译拦截 .so，并在运行环境中配置 `MEGATRACE_LOG_EXTRA_FIELDS=...,MY_ROUND,...`。

- **外部系统集成**：若将 Megatrace 作为库或子模块使用，可在自己的 .cc 中：
  1. `#include "extra_field_resolver.h"`
  2. 在 `log_writer_thread` 启动前调用 `megatrace::init_builtin_extra_resolvers()`（若尚未初始化）。
  3. 调用 `megatrace::register_extra_field_resolver("FIELD", resolver)` 注册自定义字段。

### 1.3.2 配置新字段的完整步骤（内置计算 & 非内置）

下面给出从零开始“加一个新字段”的完整流程，区分**非内置（环境变量）**与**内置计算字段**两种情况。

1. **确定字段名与含义**
   - 约定一个不会与现有字段冲突的名字，例如：`TRAIN_JOB_ID`、`CLUSTER_ID`、`MY_ROUND`。
   - 字段值最终会以 `[value]` 形式出现在日志前缀中。

2. **非内置计算字段（仅环境变量）的配置步骤**
   1. 在启动脚本 / Pod 配置中设置对应环境变量，例如：
      - `export CLUSTER_ID=cluster-a`
   2. 在同一处配置 `MEGATRACE_LOG_EXTRA_FIELDS`，将新字段名加入逗号列表，例如：
      - `export MEGATRACE_LOG_EXTRA_FIELDS=TRAIN_JOB_ID,CLUSTER_ID`
   3. 确认日志中前缀形如：
      - `[job-xxx] [cluster-a] [node_ip] [hostname] [save_count N] ...`
   4. 若某个环境变量未设置或为空，该字段对应的 `[..]` 不会输出。

3. **内置计算字段（需要代码逻辑）的配置步骤**
   1. 在 `extra_field_resolver.cc` 中新增 resolver 函数，例如：
      - `std::string resolve_my_round(const ExtraFieldContext& ctx) { /* 基于 ctx.pod_name / ctx.save_iter 等计算 */ }`
   2. 在同文件的 `init_builtin_extra_resolvers()` 中注册：
      - `register_extra_field_resolver("MY_ROUND", resolve_my_round);`
   3. 重新编译生成拦截 .so（`nccl_intercept.so` / `rccl_intercept.so`）。
   4. 在运行环境中设置：
      - `export MEGATRACE_LOG_EXTRA_FIELDS=TRAIN_JOB_ID,MY_ROUND,RUNNING_ROUND`
   5. 启动训练后，检查日志行前缀中已出现 `[MY_ROUND 的计算结果]`。

4. **混合使用示例（环境变量 + 内置字段）**
   - 启动环境：
     - `export TRAIN_JOB_ID=job-xxx`
     - `export CLUSTER_ID=cluster-a`
     - `export MEGATRACE_LOG_EXTRA_FIELDS=TRAIN_JOB_ID,CLUSTER_ID,MY_ROUND,RUNNING_ROUND`
   - 日志前缀示例：
     - `[job-xxx] [cluster-a] [<MY_ROUND>] [<RUNNING_ROUND>] [node_ip] [hostname] [save_count N] ...`

### 1.4 node_ip 获取方式

- 优先使用 **`MY_POD_IP`**（K8s 下通常由注入）。
- 若未设置或为空：使用 **本机主 IPv4**（`getifaddrs()` 取第一个非回环 IPv4 地址），适用于普通 Docker / 裸机。
- 若仍取不到：记为 `unknown`。

### 1.5 日志文件路径与命名

- **目录**：`NCCL_MEGATRACE_LOG_PATH`；若设置了 `TRAIN_JOB_ID`，则使用其子目录 `.../TRAIN_JOB_ID/`。
- **文件名**：
  - 无 job/pod 环境（无 `TRAIN_JOB_ID` 且无有效 `POD`）：`megatrace_<时间>_rank_<rank>.log`。
  - 有 job/pod：`<POD>_<时间>_<pid>.log`（或带 RUNNING_ROUND 等，由当前实现决定，此处不展开）。

---

## 二、日志行样例

### 2.1 无 extra 前缀（仅固定 node_ip、hostname）

```
[10.254.1.137] [node201] [save_count 79] [1768121799.082548236] [Rank 0] [PCI 0000:18:00.0] [Func ncclAllGather] [Data 33554432] [stream (nil)] [opCount 16932] [groupHash 0x8e5988b6aea85fa7]
```

### 2.2 有 extra 前缀（TRAIN_JOB_ID + RUNNING_ROUND）

配置：`MEGATRACE_LOG_EXTRA_FIELDS=TRAIN_JOB_ID,RUNNING_ROUND`，且设置了 `TRAIN_JOB_ID`、`POD`。

```
[job-9f34aa02-f3f9-4fcf-ab45-56f8e61baaae] [0] [10.254.1.137] [node201] [save_count 79] [1768121799.082548236] [Rank 0] [PCI 0000:18:00.0] [Func ncclAllGather] [Data 33554432] [stream (nil)] [opCount 16932] [groupHash 0x8e5988b6aea85fa7]
```

---

## 三、分析端解析（log_reader）

### 3.1 约定

- **前缀**：行首可有任意多段 `[..]`；解析时**只约定**：紧挨在 `[save_count N]` 前的两个 `[..]` 为 **node_ip**、**hostname**，其余前缀段不解析、不存储。
- **消息体**：固定格式，与写端一致（timestamp、Rank、PCI、Fun/Func、Data、stream、opCount、groupHash）。

### 3.2 正则与捕获组

`log_reader` 使用**单条正则**，结构为：

1. **可选前缀**：`(?:\[[^\]]+\]\s*)*`，匹配任意多段 `[..]`，不捕获。
2. **固定前缀**：`\[([^\]]+)\] \[([^\]]+)\] \[save_count (\d+)\]`，捕获 **node_ip**、**hostname**、**save_count**。
3. **消息体**：依次捕获 timestamp、rank、gpu_pci、function、data_size、stream、op_count、group_hash；其中 function 兼容 `[Fun xxx]` 与 `[Func xxx]`。

由此，无论可选前缀有几段（0 段、2 段、4 段等），分析端都能正确得到 node_ip、hostname 及后续字段。

### 3.3 LogEntry 字段（分析依赖）

| 字段 | 来源 | 用途简述 |
|------|------|----------|
| node_ip | 前缀倒数第 2 段 | 节点 IP，用于 metadata / 展示 |
| hostname | 前缀倒数第 1 段（save_count 前） | 主机名，用于 metadata / 展示 |
| gpu_pci | 消息体 PCI | 同上 |
| save_count | 前缀 | 按 save_count 做部分读取（最近 N 组） |
| timestamp | 消息体 | 卡顿、慢节点、时序分析 |
| rank | 消息体 | 按 rank 聚合 |
| function | 消息体 | 流分类、操作类型 |
| data_size | 消息体 | 流分类、数据量 |
| stream | 消息体 | 按 stream 分组 |
| op_count | 消息体 | 同步、group 分析 |
| group_hash | 消息体 | 并行慢、groupHash 分析 |

分析逻辑**仅依赖上述字段**；可选前缀中的 TRAIN_JOB_ID、RUNNING_ROUND 等不参与解析，仅用于存储/运维区分。

### 3.4 部分读取（save_count）

- `LogReader(max_save_count_groups=N)`：仅加载每个文件中 **最近 N 个 save_count 组** 的行（用于大文件快速分析）。
- `_extract_save_count(line)` 通过同一正则取 `save_count`，再在 `_read_recent_savecount_lines` 中按 save_count 从文件尾部逆序收集行。

---

## 四、扩展与兼容

- **写端**：在 `MEGATRACE_LOG_EXTRA_FIELDS` 中增加新名字（环境变量或内置计算字段），并保证 **node_ip、hostname 仍为最后两段**，即可扩展前缀，无需改分析。内置计算字段通过 `extra_field_resolver` 注册表扩展，与写行循环解耦。
- **分析端**：不解析、不依赖可选前缀内容；只要“最后两段 + save_count + 消息体”格式不变，解析保持兼容。
- **建议**：自定义前缀字段的值中避免包含 `]`，以免破坏 `[..]` 段解析；若必须包含，需在写端转义或与解析端约定新规则。
