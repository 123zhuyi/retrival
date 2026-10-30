---
id: kernel-nsa-selected-attention
title: NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU
type: kernel
tags: [nsa, native_sparse_attention, selected_attention, topk_gather, online_softmax, gqa, training]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1]
kernel_types: [attention]
operators: [nsa_selected_attention]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/nsa-selected-attention
related: [kernel-nsa-compress-attention-infer]
sources: [ops-transformer-nsa-selected-attention]
performance_claims: []
---

# NsaSelectedAttention（NSA 选择注意力训练前向）

CANN ops-transformer 的 `NsaSelectedAttention` 是 NSA（Native Sparse Attention）训练前向的第二阶段：根据 `topKIndices`（来自 `nsa_compress_attention_infer`）gather 选中的 KV 块，执行标准 attention 计算，输出 attentionOut + softmaxMax/Sum（供反向梯度使用）。来源见 [[ops-transformer-nsa-selected-attention]]。

## 1. 输入输出

### 输入（def.cpp L19-55；docs L68-216）

| 输入 | 必选/可选 | dtype | shape |
|---|---|---|---|
| query | 必选 | FP16/BF16 | TND `[T,N1,D_qk]` |
| key | 必选 | FP16/BF16 | TND `[T,N2,D_qk]` |
| value | 必选 | FP16/BF16 | TND `[T,N2,D_v]` |
| topkIndices | 必选 | INT32 | `[T,N2,selectedBlockCount]` |
| attenMask | 可选 | BOOL/UINT8 | - |
| actualSeqQLen | 可选（TND 必选） | INT64 | `[B]` |
| actualSeqKvLen | 可选（TND 必选） | INT64 | `[B]` |

### 输出（docs L228-255）

| 输出 | dtype | shape |
|---|---|---|
| softmaxMax | FLOAT | `[T,N1,8]` |
| softmaxSum | FLOAT | `[T,N1,8]` |
| attentionOut | FP16/BF16 | `[T,N1,D_v]` |

### 属性（def.cpp L68-79）
- scaleValue（默认1.0）、headNum、sparseMode（0=全计算/2=causal）、inputLayout（仅TND）、selectedBlockSize（≤128 16倍数）、selectedBlockCount（[1,128]）。

## 2. 核心公式

```
selected_key = Gather(key, topk_indices[i]), 0 <= i < selected_block_count
selected_value = Gather(value, topk_indices[i]), 0 <= i < selected_block_count
attention_out = Softmax(Mask(scale·Q·selected_key^T, atten_mask)) · selected_value
```

## 3. 计算流程

外层循环 S1 行 → 内层循环 N2：`GetIndices`（加载 topKIndices）→ `SelectAndGatherK`（gather 选中 K 块）→ `IterateBmm1`（QK^T，与 V gather 重叠）→ `WaitBmm1Result` → `ProcessSoftmax`（在线 softmax + mask + 输出 max/sum）→ `IterateBmm2`（PV）。

## 4. 架构支持

仅 A2 训练（Ascend910B1）√；A3 ×、推理 ×。

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。仅 A2 训练、仅 TND、D_qk=192/D_v=128 固定。Triton skeleton 标记 `reproducibility: pseudocode`。