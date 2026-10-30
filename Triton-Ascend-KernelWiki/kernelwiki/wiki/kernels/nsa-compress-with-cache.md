---
id: kernel-nsa-compress-with-cache
title: NsaCompressWithCache (NSA KV compression) on Ascend NPU
type: kernel
tags: [nsa, attention, kv_cache, compression, elementwise, reduction, aiv_only, fp16, bf16]
techniques: [fp32_accumulation, tail_mask, contiguous_access]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Kirin9030, KirinX90]
kernel_types: [attention]
operators: [nsa_compress_with_cache]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/nsa-compress-with-cache
related: [kernel-sparse-flash-attention, kernel-flash-attention-score]
sources: [ops-transformer-nsa-compress-with-cache]
performance_claims: []
---

# NsaCompressWithCache（NSA KV 压缩）

CANN ops-transformer 的 `NsaCompressWithCache` 是 **Native-Sparse-Attention 推理阶段 KV 压缩算子**：当某个 batch 的 token 数凑满一个 `compressBlockSize` 窗口时，将该 batch 后 `compressBlockSize` 个 token 用 `weight` 做加权求和压缩成一个 `compress_token`，通过 `slot_mapping` 写回 `output_cache`（README L17-23）。它是 NSA 稀疏注意力流水线的上游步骤，压缩后的 KV 供下游稀疏 attention 使用。

## 1. 输入输出

### 输入（def.cpp L21-51）

| 输入 | 必选/可选 | 含义 | dtype | format |
|---|---|---|---|---|
| input | 必选 | 原始的 KV cache | FP16/BF16 | ND |
| weight | 必选 | 压缩权重 | FP16/BF16 | ND |
| slot_mapping | 必选 | 每个 batch 压缩结果存储位置索引 | INT32 | ND |
| output_cache | 必选（输入/输出同 tensor） | 输出的压缩后 cache | FP16/BF16 | ND |
| act_seq_len | 可选 | 实际序列长度 | INT64 | ND |
| block_table | 可选 | PageAttention 块表 | INT32 | ND |

### 输出（def.cpp L52-56）
- `output_cache`（REQUIRED，原地更新）

### 属性（def.cpp L57-61）
`layout`("TND")、`compress_block_size`(必选)、`compress_stride`(必选)、`act_seq_len_type`(必选)、`page_block_size`(必选)。

### 约束（README L93-105）
- compressBlockSize/stride 16 倍数，compressBlockSize≤64，compressBlockSize≥stride。
- headDim 16 倍数，headDim≤256；headNum≤64（>50 时偶数）。
- pageBlockSize=64 或 128；layout 仅 TND（属性值不生效）。
- 不支持 input/weight/output_cache 为空。

## 2. 核心公式

```
compressIdx = (seqLen - compressBlockSize) / stride
outputCache[slot_mapping[i]] = Σ(input[compressIdx*stride : compressIdx*stride+compressBlockSize] × weight[:])
```
（README L28-29）。weight 与 input 按第三维 broadcast。

## 3. 计算流程

| 阶段 | 操作 | 来源 |
|---|---|---|
| 1. InitBase | 读 tiling data，算每核 batch/head 范围，分配 ping-pong buffer | kernel L38-64 |
| 2. InitActSeqLen | 遍历 batch，检查 `(seqLen-compressBlockSize)%stride==0` | kernel L66-78 |
| 3. InitWeight | weight 搬入 UB，Cast to FP32 | kernel L80-91 |
| 4. Compress | 滑动窗口内 token 加权求和 → scatter store | Process |

## 4. tiling 与分核（详见 `artifacts/nsa-compress-with-cache/tiling_dispatch_tables.md`）

- 分核：按 head×batch 切分（`coreStartBatchIdxList`/`coreStartHeadIdxList`）。
- 核内：`headsNumPerCore`、`tokenNumPerTile`/`tokenRepeatCount`、`bufferNum` ping-pong。
- 架构：A2/A3（FP16/BF16）+ KirinX90/Kirin9030（仅 FP16）。

## 5. 架构支持（README L7-14；def.cpp L62-67）

| 产品 | 架构 | 支持 | dtype |
|---|---|---|---|
| Atlas A2/A3 训练/推理 | Ascend910B1/910_93 | √ | FP16/BF16 |
| Kirin X90 处理器系列 | KirinX90 | √ | FP16 |
| Kirin 9030 处理器系列 | Kirin9030 | √ | FP16 |

不支持 950/310P/训练系列。

## 6. Agent actions

1. 识别为 NSA KV 压缩（推理阶段滑动窗口加权求和 + scatter 写回）。
2. 按 batch 遍历，检查 `(seqLen-compressBlockSize)%stride==0` 触发压缩。
3. 滑动窗口 offset 计算：`start = compressIdx * stride`，长度 `compressBlockSize`。
4. weight broadcast 乘后 `tl.sum(axis=0)` 归约。
5. slot_mapping scatter 写回 output_cache。
6. PA 场景需通过 block_table 查物理 page 偏移。
7. `compressBlockSize` 和 `stride` 为 16 倍数，`compressBlockSize≤64`。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、weight 初始化方式（`weightInitFlag`）未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-nsa-compress-with-cache]]。