---
id: kernel-lightning-indexer
title: LightningIndexer (sparse attention indexer) on Ascend NPU
type: kernel
tags: [lightning_indexer, sparse_attention, indexer, topk, relu, gqa, page_attention]
techniques: [fp32_accumulation, sparse_kv_gather]
hardware_features: [unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [lightning_indexer]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/lightning-indexer
related: [kernel-sparse-flash-attention, technique-sparse-kv-gather]
sources: [ops-transformer-lightning-indexer]
performance_claims: []
---

# LightningIndexer（稀疏注意力索引选择器）

CANN ops-transformer 的 `LightningIndexer` 是稀疏注意力（SparseFlashAttention）的**索引选择器**：根据 Q_index 和 K_index 计算重要性分数，经 ReLU 激活和 weights 缩放后，topK 选出每个 token 对应的最重要 KV 位置（sparse_indices），供下游稀疏注意力算子使用。来源见 [[ops-transformer-lightning-indexer]]。

## 1. 输入输出

### 输入（def.cpp L23-37；README L39-53）

| 输入 | 必选/可选 | dtype | shape |
|---|---|---|---|
| query | 必选 | FP16/BF16 | BSND`[B,S1,N1,D]`/TND`[T1,N1,D]`，N1≤64 |
| key | 必选 | FP16/BF16 | BSND`[B,S2,N2,D]`/TND`[T2,N2,D]`/PA_BSND`[block_count,block_size,N2,D]`，N2=1 |
| weights | 必选 | FP16/BF16/FP32 | `[B,S1,N1]`/`[T1,N1]` |
| actual_seq_lengths_q | 可选 | INT32 | `[B]`，TND 必选 |
| actual_seq_lengths_k | 可选 | INT32 | `[B]`，TND/PA 必选 |
| block_table | 可选 | INT32 | `[B, maxBlockNumPerSeq]`，PA 场景 |

### 输出（README L74-77）

| 输出 | dtype | shape |
|---|---|---|
| sparse_indices | INT32 | `[B,S1,N2,sparseCount]`/`[T1,N2,sparseCount]` |
| sparse_values | FP16/BF16 | 同 sparse_indices shape（可选，return_values=True） |

### 属性（def.cpp L61-67）
- layout_query（BSND/TND）、layout_key（BSND/TND/PA_BSND）、sparse_count（2048，最高 8192）、sparse_mode（0/3 rightDownCausal）、pre_tokens/next_tokens（仅 INT64_MAX）、return_values（False）。

## 2. 核心公式

`Indices = Top-k( [1]_{1×g} @ [(W @ [1]_{1×S_k}) ⊙ ReLU(Q_index @ K_index^T)] )`

即：Q·K^T → ReLU（负数置 0）→ weights 缩放 → g 组求和 → topK 选择最重要的 sparse_count 个位置。

## 3. 架构支持

A2（Ascend910B1）√、A3（Ascend910_93）√；训练+推理。

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。