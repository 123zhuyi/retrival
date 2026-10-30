---
id: kernel-nsa-compress-attention-infer
title: NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU
type: kernel
tags: [nsa, native_sparse_attention, compress_attention, sliding_window, importance_score, topk, paged_attention, gqa, cube_vector_pipe]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [nsa_compress_attention_infer]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/nsa-compress-attention-infer
related: []
sources: [ops-transformer-nsa-compress-attention-infer]
performance_claims: []
---

# NsaCompressAttentionInfer（NSA 压缩注意力推理）

CANN ops-transformer 的 `NsaCompressAttentionInfer` 是 NSA（Native Sparse Attention）推理流水线的第一阶段：在压缩后的 KV 上执行注意力计算，产出重要性分数（滑动窗口卷积），最终选择 topK 个最重要 KV 块索引，供后续 `nsa_selected_attention_infer` 使用。来源见 [[ops-transformer-nsa-compress-attention-infer]]。

## 1. 输入输出

### 输入（def.cpp L19-56；docs L77-251）

| 输入 | 必选/可选 | 含义 | dtype | shape |
|---|---|---|---|---|
| query | 必选 | Q | FP16/BF16 | TND `[T,N1,D]` / BSND `[B,N1,S,D]`；S≤4[多token] |
| key | 必选 | K（PagedAttention） | FP16/BF16 | `[numBlocks, blockSize, N2*D_QK]` |
| value | 必选 | V（PagedAttention） | FP16/BF16 | `[numBlocks, blockSize, N2*D_V]` |
| blockTable | 必选 | block 映射表 | INT32 | `[B, maxBlockNumPerSeq]` |
| actualCmpKvSeqLen | 必选 | 压缩后 KV 实际长 | INT64 | `[B]` |
| actualQSeqLen | 可选 | Q 实际长 | INT64 | `[B]` |
| actualSelKvSeqLen | 可选 | 压缩前 KV 实际长 | INT64 | `[B]` |
| attentionMask | 可选 | 注意力掩码 | - | Q_S>1 时生效 |
| topKMask | 可选 | topK 掩码（预留） | - | - |

### 输出（def.cpp L57-63）

| 输出 | 含义 | dtype | shape |
|---|---|---|---|
| output | 注意力输出 | FP16/BF16 | 同 query |
| topKOutput | topK 块索引 | INT32 | `[B, N2, selectBlockCount]` 或 `[T, N2, selectBlockCount]` |

### 属性（def.cpp L64-86）
- numHeads（N1）、numKeyValueHeads（N2）、selectBlockSize（16-128，≥compressBlockSize，且 compressBlockStride 整数倍）、selectBlockCount、compressBlockSize（16-128）、compressBlockStride（16-64）、scaleValue、layoutOptional（TND/BSND）、pageBlockSize（≤128，16 倍数）、sparseMode。

## 2. 核心计算公式

```
P_cmp = Softmax(scale · Q · K^T)
attentionOut = P_cmp · V
P_slc[j] = sum_{m=0}^{l'/d-1} sum_{n=0}^{l/d-1} P_cmp[l'/d · j - m - n]   (滑动窗口卷积)
P_slc' = sum_{g=1}^{G} P_slc^g                                            (G = groupSize)
topKIndices = topk(P_slc')
```

## 3. 计算流程（Cube/Vector 分离流水）

Cube（AIC）路径：`ProcessMm1`（QK^T，通过 blockTable 映射 KV 块）→ cross-core sync → `ProcessMm2`（PV）。
Vector（AIV）路径：wait QK^T → `ProcessSoftmax`（含 mask）→ sync → `ProcessImportanceScore`（滑动窗口三角形权重卷积）→ `ProcessTopK`（输出 int32 索引）。

## 4. 架构支持

A2 训练+推理 √、A3 训练+推理 √；其余 ×。

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。滑动窗口卷积权重细节、tiling 数值等需硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。