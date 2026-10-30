---
id: technique-sparse-attention-tiling
title: SparseFlashAttention tiling and core-partition strategy on Ascend NPU
type: technique
tags: [tiling, sparse_attention, flash_attention, c_template, v_template]
techniques: [sparse_attention_tiling]
hardware_features: [unified_buffer, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [sparse_flash_attention]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: snippet
sources: [ops-transformer-sparse-flash-attention]
prerequisites: [了解FlashAttention基本算法, CUBE/VECTOR计算单元分工, SparseAttention与稀疏索引(TopK)基本概念]
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline, pattern-flash-attention-tiling-key-dispatch]
indicators: [indicator-02, indicator-05, indicator-09]
---

# SparseFlashAttention tiling 与分核策略

SparseFlashAttention（SFA）是 FlashAttention 在稀疏注意力场景下的变体。核心区别在于：S2（KV 侧）不再连续遍历，而是由 **sparse_indices（TopK）** 驱动的离散 block 遍历。这一差异导致 tiling 策略与标准 FA 有本质不同：SFA 只有 2 类模板（C_TEMPLATE/V_TEMPLATE），而 FA 有 6 类；SFA 的 headDim 固定为 512+64=576（MLA 场景）；SFA 的 S2 支持 FlashDecoding 核间切分。

## 1. 与 FlashAttention tiling 的差异

| 维度 | FA | SFA |
|------|----|-----|
| S2 遍历 | 连续 S2 切分 | sparse_indices 驱动的 block 遍历 |
| 模板数 | 6 类（arch32） | 2 类（C_TEMPLATE/V_TEMPLATE） |
| headDim | 可变（1~768） | 固定 512+64=576（MLA 场景） |
| 核间切分 | S2 不切到多核 | S2 支持 FlashDecoding 核间切分 |
| 稀疏索引 | 无 | TopK sparse_indices 输入 |

## ️2. Tiling 数据结构

`SparseFlashAttentionTilingDataMla`（tiling.h:196-203）包含 5 个子结构：

### 2.1 基础参数（BaseParams）

```
- batchSize, seqSize, qSeqSize
- blockSize（PA 场景 page block 大小）
- maxBlockNumPerBatch（PA 场景最大 block 数）
- scaleValue, nNumOfQInOneGroup
- actualLenDimsQ, actualLenDimsKV（TND 场景）
- outputLayout, sparseMode
- preTokens, nextTokens（band mask 参数）
- attentionMode
- sparseBlockSize（稀疏 block 大小，单位元素）
- sparseBlockCount（稀疏 block 总数）
```

### 2.2 SplitKV 参数（splitKVParams）

```
- s2: S2 切分份数（FlashDecoding 场景）
- accumOutSize: FlashDecoding workspace 大小
- logSumExpSize: FlashDecoding LSE workspace 大小
```

### 2.3 单核参数（singleCoreParams）

```
- usedCoreNum: 实际使用的核数
```

### 2.4 单核 Tensor 大小（singleCoreTensorSize）

```
- mmResUbSize: Mm1（QK）结果 UB 大小
- bmm2ResUbSize: Mm2（PV）结果 UB 大小
```

### 2.5 内切基本块参数（innerSplitParams）

```
- mBaseSize: S1 方向基本块，默认 128
- s2BaseSize: S2 方向基本块，由 tiling 动态计算
```

## 3. C_TEMPLATE vs V_TEMPLATE 双模板

`TEMPLATE_MODE` 控制（template_tiling_key.h:27-28），由 `InitParams` 在 host 侧决定（tiling.cpp:276-281）：

### 3.1 选择逻辑

```cpp
if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) {
    perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
} else {
    perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
}
```

当 `sparseBlockSize ≤ 4` 且 `s2Size ≠ 0` 时走 V_TEMPLATE，否则走 C_TEMPLATE。

### 3.2 C_TEMPLATE（Cube 主）

| 阶段 | 执行单元 | 计算内容 |
|------|---------|---------|
| Mm1 | Cube（AIC） | Q × K^T，输出 mmRes（FP32） |
| Softmax | Vector（AIV） | FlashSoftmax + mask + scale |
| Mm2 | Cube（AIC） | softmax_out × V，输出 bmm2Res |

C_TEMPLATE 中 Cube 承担主要计算量，Vector 仅处理 softmax。Cube 和 Vector 之间通过核间同步 Event 协作。

### 3.3 V_TEMPLATE（Vector 主）

| 阶段 | 执行单元 | 计算内容 |
|------|---------|---------|
| MergeKv | Vector（AIV） | 离散 KV 合并（sparse block → 连续 buffer） |
| Mm1 | Cube（AIC） | Q × K^T |
| Softmax | Vector（AIV） | FlashSoftmax |
| Mm2 | Cube（AIC） | softmax_out × V |
| 输出处理 | Vector（AIV） | 结果写回 |

V_TEMPLATE 中 Vector 额外负责 MergeKv：将离散的 sparse block 数据从 GM 搬运到连续的 UB buffer 中，供后续 Cube 计算。当 `sparseBlockSize` 很小时（≤4），离散 block 数量多，Vector 的 MergeKv 开销占比增大，但整体仍优于 Cube 逐个搬运。

## 4. 稀疏索引驱动的 S2 遍历

这是 SFA 与标准 FA 最核心的差异。标准 FA 的 S2 循环按连续区间切分；SFA 的 S2 循环由 `CalcSinnerTopKBegin`（kernel_mla.h:941-1016）控制。

### 4.1 遍历流程

```
从 topKGm（sparse_indices）读取 block 索引
        │
        ▼
计算有效 block 数：validCount = min(sparseBlockCount, threshold / sparseBlockSize)
        │
        ▼
依次读取 topKIdx 位置的 sparse_indices 值
        │
        ├── 值为 -1 → 无效 block，跳过
        │
        ▼
blockBegin = sparseIndices * sparseBlockSize
blockLen = min(blockBegin + sparseBlockSize, thresold) - blockBegin
        │
        ▼
累积 sparseLen，直到 >= s2BaseSize 或遍历完所有 block
        │
        ▼
累积达到 s2BaseSize → 聚合为一个连续 chunk 供 Cube 计算
最后一次不足 s2BaseSize → 作为 tail 处理
```

### 4.2 关键特性

- **Block 粒度**：每个 sparse block 的大小为 `sparseBlockSize`（由属性传入）
- **Chunk 聚合**：将离散的 block 累积为 `s2BaseSize` 大小的连续 chunk
- **阈值截断**：有效 block 超过 `thresold = nextTokens + preTokens` 时截断
- **Tail 处理**：最后一个 chunk 可能小于 `s2BaseSize`，存入 `s2BasicSizeTail`

### 4.3 与连续 S2 对比

| 特性 | 连续 S2 | 稀疏 S2 |
|------|---------|---------|
| 循环边界 | 等间隔 `s2BaseSize` | 不等间隔，取决于 block 分布 |
| 访存模式 | 连续 | 离散（GM 随机读取） |
| 数据量 | 固定 | 取决于 sparse 密度 |
| 负载预测 | 确定 | 需通过 block 数估计 |

## 5. 基本块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| mBaseSize | 128 | S1（Q 侧）方向基本块大小 |
| mFdBaseSize | 8 | FlashDecoding 场景的 S1 基本块 |
| nBufferMBaseSize | 256 | N buffer 基本块大小（kernel 编译期常量） |
| s2BaseSize | 动态计算 | S2 方向基本块，由 tiling 动态计算 |
| PRELOAD_NUM | 2 | 预加载流水深度 |
| headDimAlign | Align(qkHeadDim, 32B) | headDim 按 32 字节对齐 |

`mBaseSize` 用于 gS1 轴的循环切分：`gS1SplitNum = (actS1Size * gSize + mBaseSize - 1) / mBaseSize`。

## 6. 核间切分（ProcessBalance）

`ProcessBalance`（kernel_mla.h:785-849）按 **B/N2/G/S1** 四轴做负载均衡。

### 6.1 分核方式

```
for bN2LoopIdx = bN2Start → bN2End:      // B/N2 轴分配
  GetBN2Idx(bN2LoopIdx) → bIdx, n2Idx
  GetActualSeqLen(bIdx)                   // 获取实际序列长度
  for gS1LoopIdx = gS1Start → gS1End:    // G/S1 轴分配
    CalcSinnerTopKBegin(...)               // 稀疏索引遍历
    for s2LoopIdx = s2Start → s2LoopTimes:
      PreloadPipeline(...)                 // 三级流水
```

### 6.2 分核范围

每核通过 `constInfo.bN2Start/bN2End/gS1Start/gS1End` 确定自己负责的范围。由 host 侧 `Split` / `SplitBalanced` 计算，确保：

- **多核用满**：所有可用的 AIC 核都参与计算
- **负载均匀**：B/N2/G/S1 轴上的任务量尽可能均衡
- **FlashDecoding 支持**：`splitKVFlag_` 为 true 时，S2 轴也参与核间切分

### 6.3 FlashDecoding S2 核间切分

当 S2 很大且需要多核并行处理时，SFA 支持将 S2 轴切分到多个核：

- `splitKVNum`：S2 切分份数
- `coreStartKVSplitPos`：当前核的 KV 起始位置
- `tndIsS2SplitCore`：当前核是否做 S2 切分

这与标准 FA 的 **"S2 不切到多核"** 原则不同，因为 SFA 的 sparse 场景下，S2 切分不影响 FlashSoftmax 的正确性（离散 block 的 softmax 可以独立计算后合并）。

## 7. UB 预算计算

### 7.1 CalcUbBmm（tiling.cpp:289-299）

```cpp
void SFAMlaTiling::CalcUbBmm() {
    uint32_t cubeMSize = min(sfaInfo_->gSize * sfaInfo_->s1Size, mBaseSize_);
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);
    qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}
```

### 7.2 MLA 场景的 headDim

- headDim（nope）：512
- headDimRope（rope）：64
- **合计**：576

### 7.3 UB 分配约束

`CheckUbSpace` 确保：

```
mmResUb + bmm2ResUb + vec1ResUb + softmax临时缓冲区 ≤ ubSize
```

如果 UB 不足，需要缩小 `sInnerSize` 或 `mBaseSize`。

### 7.4 关键 UB 区域

| UB 区域 | 内容 |
|---------|------|
| mmResUb | Mm1 输出（QK^T 结果，FP32） |
| bmm2ResUb | Mm2 输出（PV 结果，FP32） |
| vec1ResUb | Vector 处理 softmax 中间结果 |
| qPreSizeMla | Q 数据预取区域（含 rope）|

## 8. 与 FA tiling 的关联

SFA 的 tiling 策略借鉴了 FA 的核内/核间切分原则，包括：
- **核内切分**：使用 `mBaseSize`/`s2BaseSize` 将大矩阵切分为适合 UB 的小块
- **CV 协作**：Cube 负责 matmul，Vector 负责 softmax/merge
- **Workspace 分配**：预留 workspace 用于跨核数据交换

但稀疏索引驱动的 S2 遍历使 SFA 的 tiling 逻辑完全不同：
- 无需 FA 的 6 类模板选择，只用 2 类（C_TEMPLATE/V_TEMPLATE）
- S2 循环由 `sparse_indices` 动态驱动，非连续区间
- 支持 FlashDecoding 的 S2 核间切分（FA 不支持）

详见 [[technique-flash-attention-tiling]] 了解 FA 的通用 tiling 框架。

## 9. Agent actions

1. 确认 sparseBlockSize 和 s2Size，选择 C_TEMPLATE 或 V_TEMPLATE。
2. 从 sparse_indices 读取 TopK block 索引，按 block 粒度遍历 S2。
3. 将离散 block 聚合为 s2BaseSize 大小的连续 chunk 供 Cube 计算。
4. 计算 UB 大小：mmResUb + bmm2ResUb + softmax 临时 ≤ ubSize。
5. 按 B/N2/G/S1 四轴做核间负载均衡（ProcessBalance）。
6. 若 FlashDecoding 场景，额外支持 S2 核间切分（splitKVNum）。
7. 设置 tilingKey 标识 TEMPLATE_MODE，下发到 kernel。

来源与详细分析见 [[ops-transformer-sparse-flash-attention]]。