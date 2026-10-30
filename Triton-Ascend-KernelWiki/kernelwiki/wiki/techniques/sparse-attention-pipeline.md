---
id: technique-sparse-attention-pipeline
title: SparseFlashAttention Cube-Vector pipeline orchestration on Ascend NPU
type: technique
tags: [pipeline, sparse_attention, flash_attention, cube_vector_pipe, preload, double_buffer]
techniques: [sparse_attention_pipeline, fp32_accumulation]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [sparse_flash_attention]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: snippet
sources: [ops-transformer-sparse-flash-attention]
related: [technique-flash-attention-pipeline, technique-sparse-attention-tiling, pattern-flash-attention-tiling-key-dispatch]
prerequisites: [了解FlashAttention四阶段计算流程, CV流水基本概念, SparseAttention基本算法]
indicators: [indicator-06, indicator-08, indicator-12, indicator-17]
---

# SparseFlashAttention 流水编排

SparseFlashAttention (SFA) 在标准 FlashAttention 的四阶段流水基础上，额外增加了 MergeKv 阶段（V_TEMPLATE 模式），形成 **5 阶段 CV 流水**。流水采用 3-slot preload pipeline 协调 Cube 与 Vector 的计算，Vector 侧使用 ping-pong double buffer 实现数据搬运与计算重叠。

## 1. 五阶段计算流水

SFA 的计算流水分为 5 个阶段，Cube (AIC) 与 Vector (AIV) 交替执行：

| 阶段 | 计算单元 | 入口函数 | 操作 | 数据依赖 |
|------|----------|----------|------|---------|
| Stage 0: MergeKv (V_TEMPLATE only) | Vector (AIV) | `MergeKv` | 根据 `sparse_indices` 从 GM 离散读取 KV 并合并到连续 buffer | 依赖 `sparse_indices` 和 KV cache |
| Stage 1: Bmm1 | Cube (AIC) | `ComputeMm1` (kernel_mla.h:729) | Q·Kᵀ matmul | 依赖 MergeKv 输出的连续 K |
| Stage 2: Vec1 | Vector (AIV) | `ProcessVec1L` | FlashSoftmax（在线 max/sum/exp） | 依赖 Bmm1 输出 |
| Stage 3: Bmm2 | Cube (AIC) | `ComputeMm2` (kernel_mla.h:743) | P·V matmul | 依赖 Vec1 的 softmax 结果 + MergeKv 输出的连续 V |
| Stage 4: Vec2 | Vector (AIV) | `ProcessVec2L` | rescale → 写回 attention_out | 依赖 Bmm2 输出 |

**关键依赖链**：`MergeKv → Bmm1 → Vec1 → Bmm2 → Vec2`。与标准 FA 相比，MergeKv 是 SFA 独有的前置阶段。

**TEMPLATE_MODE 决定**（SFAT 模板参数 `TEMPLATE_MODE`，common.h:36）：
- **C_TEMPLATE**：Cube 做 Mm1+Mm2，Vector 做 Vec1+Vec2。无 MergeKv 阶段。
- **V_TEMPLATE**：Vector 额外负责 MergeKv（Stage 0）。Cube 等待 `syncV0C1` 信号后才开始 Mm1。

## 2. PreloadPipeline 三缓冲

`PreloadPipeline`（kernel_mla.h:852-894）是 SFA 的核心流水编排函数，使用 **3 个 slot 的循环缓冲**：

```cpp
static constexpr uint32_t PRELOAD_NUM = 2;                   // kernel_mla.h:84
static constexpr uint32_t SFA_PRELOAD_TASK_CACHE_SIZE = 3;   // kernel_mla.h:86
static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;    // kernel_mla.h:107
```

### Slot 分配

```
extraInfo[loop % 3]     → 本轮任务（Mm1 / MergeKv）
extraInfo[(loop+1) % 3] → 上两轮任务（Vec2 消费）
extraInfo[(loop+2) % 3] → 上一轮任务（Mm2 消费）
```

### 流水时间轴

```
         loop=0         loop=1         loop=2         loop=3         loop=4
Slot0: | Mm1(V0)     | Vec1(V1)     | Mm2(C2)      | Vec2(V2)     |
Slot1: |              | Mm1(V0)      | Vec1(V1)     | Mm2(C2)      | Vec2(V2)     |
Slot2: |              |              | Mm1(V0)      | Vec1(V1)     | Mm2(C2)      | Vec2(V2)     |
```

- **loop 0**：Cube 发射 Mm1（slot 0），Vector 空闲。
- **loop 1**：Vector 消费 slot 2（无效），Cube 发射 Mm1（slot 1）。
- **loop 2**：Vector 消费 slot 0（Vec1），Cube 消费 slot 2（Mm2），同时发射 Mm1（slot 2）。**首次达到全流水**。
- **loop ≥ 3**：流水稳定后，每轮 Vector 处理 Vec1（上一轮结果）+ Vec2（上两轮结果），Cube 处理 Mm2（上一轮结果）+ Mm1（本轮）。

### PreloadPipeline 核心逻辑（kernel_mla.h:852-894）

```cpp
RunInfo &extraInfo0 = extraInfo[loop % SFA_PRELOAD_TASK_CACHE_SIZE];         // 本轮
RunInfo &extraInfo2 = extraInfo[(loop + 2) % SFA_PRELOAD_TASK_CACHE_SIZE];   // 上一轮（Vec1 消费）
RunInfo &extraInfo1 = extraInfo[(loop + 1) % SFA_PRELOAD_TASK_CACHE_SIZE];   // 上两轮（Mm2 消费）

CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);  // 计算本轮参数

if (extraInfo0.isValid) {
    if ASCEND_IS_AIC {
        // Cube: Mm1 (等待 V0 完成 MergeKv)
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) CrossCoreWaitFlag(syncV0C1);
        ComputeMm1(extraInfo0);
    } else {
        // Vector: MergeKv (V_TEMPLATE)
        if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
            vectorService.MergeKv(extraInfo0);
            CrossCoreSetFlag(syncV0C1);  // 通知 Cube 可以开始 Mm1
        }
    }
}
if (extraInfo2.isValid) {
    if ASCEND_IS_AIV { vectorService.ProcessVec1L(extraInfo2); }  // Vector: Vec1
    if ASCEND_IS_AIC { ComputeMm2(extraInfo2); }                    // Cube: Mm2
}
if (extraInfo1.isValid) {
    if ASCEND_IS_AIV { vectorService.ProcessVec2L(extraInfo1); }  // Vector: Vec2
    extraInfo1.isValid = false;
}
```

## 3. CV 间 event 同步

SFA 使用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`（kernel_mla.h:30-31）实现 Cube 与 Vector 之间的硬件级同步。同步事件定义在 `ConstInfo` 中（common.h:137-142）：

| Event | 语义 | 设置方 | 等待方 |
|-------|------|--------|--------|
| `syncV0C1` | Vector0 (MergeKv) 完成 → Cube1 (Mm1) 开始 | Vector（V_TEMPLATE） | Cube |
| `syncC1V1` | Cube1 (Mm1) 完成 → Vector1 (Vec1) 开始 | Cube（ComputeMm1, kernel_mla.h:738） | Vector |
| `syncV1C2` | Vector1 (Vec1) 完成 → Cube2 (Mm2) 开始 | Vector | Cube（ComputeMm2, kernel_mla.h:751） |
| `syncC2V2` | Cube2 (Mm2) 完成 → Vector2 (Vec2) 开始 | Cube（ComputeMm2, kernel_mla.h:753） | Vector |
| `syncC2V1` | Cube2 (Mm2) 完成 → Vector1（下一轮 Vec1）开始 | Cube（ComputeMm2, kernel_mla.h:754） | Vector |

### 同步常量（kernel_mla.h:88-93）

```cpp
static constexpr uint32_t SYNC_V0_C1_FLAG = 6;    // V0 → C1
static constexpr uint32_t SYNC_C1_V1_FLAG = 7;    // C1 → V1
static constexpr uint32_t SYNC_V1_C2_FLAG = 8;    // V1 → C2
static constexpr uint32_t SYNC_C2_V2_FLAG = 9;    // C2 → V2
static constexpr uint32_t SYNC_C2_V1_FLAG = 4;    // C2 → V1 (next round)
```

### V_TEMPLATE 额外同步

V_TEMPLATE 模式下，Cube 侧在 `ProcessBalance`（kernel_mla.h:792-798）开头额外设置 4 次 `PIPE_MTE2` flag（event 3），Vector 侧在结束时对应等待 4 次（kernel_mla.h:841-847）。这用于 **MergeKv 阶段的 Cube-Vector 对齐**。

## 4. Vector 侧 double buffer

Vector 侧使用 **ping-pong double buffer** 实现数据搬运与计算重叠。`InitBuffers`（vector_mla.h:211-248）中分配：

```cpp
pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2);  // 2:pingpong
pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K * 2);   // 2:pingpong
pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);
```

- `inputBuff1` / `inputBuff2` 使用 `* 2` 分配双倍空间，即 ping-pong 两片。
- preload 相关 buffer（`nValueBuff`, `cofValueBuff`, `aMlaSumBuff`, `softmaxMaxBuff`, `softmaxExpBuff`, `softmaxSumBuff`）均按 `constInfo.preLoadNum` 倍数分配（vector_mla.h:222-228），支持多轮 preload 的中间结果保留。

MergeKv 阶段使用 `MERGE_CACHE_GM_BUF_NUM = 4`（vector_mla.h:135）的循环 buffer 管理离散 KV 搬运。

## 5. Cube 侧 L0/L1 ping-pong

Cube 侧 `InitBuffers`（cube_mla.h:301-317）同样使用 ping-pong buffer：

```cpp
pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);  // L0A ping-pong
pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);  // L0B ping-pong
pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);  // L0C ping-pong
pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);   // QP L1 四缓冲
pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);   // KV L1 三缓冲
```

- L0A/L0B/L0C 各使用 **2 块** ping-pong 切换，让 MTE1（数据搬运）与 MTET（计算）重叠。
- QP L1 使用 **4 块** buffer，KV L1 使用 **3 块** buffer，对应不同数据流的搬运深度。

## 6. Workspace 内存布局

Init（kernel_mla.h:463-489）中 workspace 按以下顺序排列：

```
| Q (per-core) | mm1ResGm (S 矩阵) | vec1ResGm (P 矩阵) | mm2ResGm (O) | [kvMergeGm] (V_TEMPLATE) |
```

每个中间结果按 `dbWorkspaceRatio = PRELOAD_NUM = 2` 倍分配，支持双缓冲（kernel_mla.h:469-479）：

```cpp
mm1ResGm.SetGlobalBuffer(
    (__gm__ MM1_OUT_T *)(workspace + offset +
        aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
```

V_TEMPLATE 额外分配 `kvMergeGm_`（512 × 576 × 4 元素）和 `kvValidSizeGm_`（kernel_mla.h:483-488），用于存放 MergeKv 合并后的连续 KV 数据。

## 7. 与 FA 流水对比

| 差异点 | FA | SFA |
|--------|----|-----|
| 阶段数 | 4（Bmm1→Vec1→Bmm2→Vec2） | 5（MergeKv→Bmm1→Vec1→Bmm2→Vec2） |
| 额外流水 | 无 | V_TEMPLATE 下 MergeKv 处理离散 KV |
| preload 深度 | 2（双发）或 3（950PR） | 2（`PRELOAD_NUM=2`） |
| task cache size | 3（loop % 3） | 3（`SFA_PRELOAD_TASK_CACHE_SIZE=3`） |
| 数据依赖 | Bmm1→Vec1→Bmm2→Vec2 链式 | 增加 MergeKv→Bmm1 依赖 |
| 同步事件 | syncC1V1 / syncV1C2 / syncC2V2 | 额外 syncV0C1 / syncC2V1 |

SFA 与 FA 的核心差异在于 **MergeKv 前置阶段**。SFA 需要从离散的 KV cache 中按 `sparse_indices` 索引读取数据并合并为连续块，这个操作在 Vector 侧完成（V_TEMPLATE），通过 `syncV0C1` 与后续的 Cube Mm1 同步。

## 8. 编程视角

### 8.1 Init 流程（kernel_mla.h:412-521）

1. 初始化 tiling data（`InitTilingData`）
2. 初始化 actual sequence lengths
3. 初始化全局 tensor 指针（`queryGm`, `keyGm`, `valueGm`, `sparseIndices` 等）
4. 分配 workspace 内存（`mm1ResGm`, `vec1ResGm`, `mm2ResGm`, `kvMergeGm_`）
5. 根据 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 分支初始化 `matmulService` 或 `vectorService`：
   - AIC: `matmulService.InitParams` / `InitMm1GlobalTensor` / `InitMm2GlobalTensor`
   - AIV: `vectorService.InitParams` / `InitVec0GlobalTensor` (V_TEMPLATE) / `InitVec1GlobalTensor` / `InitVec2GlobalTensor`
6. 调用 `InitBuffers` 分配 UB/L1 缓冲区

### 8.2 Process 入口（kernel_mla.h:758-775）

```cpp
void Process() {
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            vectorService.InitSoftmaxDefaultBuffer();
        } else {
            matmulService.AllocEventID();
        }
        ProcessBalance();  // 主循环：遍历 bN2/gS1/s2 轴，调用 PreloadPipeline
        // 释放 event ID
    }
}
```

### 8.3 关键参数

| 参数 | 值 | 含义 |
|------|----|------|
| `PRELOAD_NUM` | 2 | Cube preload 深度（双发） |
| `SFA_PRELOAD_TASK_CACHE_SIZE` | 3 | 循环 buffer slot 数 |
| `N_BUFFER_M_BASIC_SIZE` | 256 | M 轴基本块大小 |
| `MERGE_CACHE_GM_BUF_NUM` | 4 | MergeKv GM 缓存 buffer 数（V_TEMPLATE） |

## 9. Agent actions

1. 确认 TEMPLATE_MODE 后选择流水策略：C_TEMPLATE（4 阶段） vs V_TEMPLATE（5 阶段含 MergeKv）。
2. 设置 preload 深度：默认 `PRELOAD_NUM=2`，对应 3-slot 循环 buffer。
3. 分配 workspace 时按 `dbWorkspaceRatio = PRELOAD_NUM` 倍预留中间结果空间。
4. V_TEMPLATE 下确保 MergeKv 的 `kvMergeGm_` 和 `kvValidSizeGm_` workspace 充足。
5. 配置 5 个 sync event（`syncV0C1`, `syncC1V1`, `syncV1C2`, `syncC2V2`, `syncC2V1`），避免 CV 间数据竞争。
6. Vector 侧 UB buffer 按 ping-pong 分配（`* 2`），preload 相关 buffer 按 `preLoadNum` 倍数分配。

详见 [[technique-flash-attention-pipeline]] 对比 FA 流水，[[pattern-flash-attention-tiling-key-dispatch]] 了解 SFA 的 key dispatch 策略，源码见 [[ops-transformer-sparse-flash-attention]]。