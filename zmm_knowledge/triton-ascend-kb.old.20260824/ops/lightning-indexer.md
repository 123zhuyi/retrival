---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: LightningIndexer (sparse attention indexer) on Ascend NPU
description: CANN ops-transformer 的 LightningIndexer 是稀疏注意力（SparseFlashAttention）的索引选择器：根据 Qindex 和 Kindex 计算重要性分数，经 ReLU 激活和 weights 缩放后，topK 选出每个 token 对应的最重要 KV 位置（sparseindices），供下游稀疏注意力算子使用。来源见 ops-transformer-lightning-indexer。
tags: [lightning_indexer, sparse_attention, indexer, topk, relu, gqa, page_attention, fp32_accumulation, sparse_kv_gather, unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2, ascend910b1, ascend910_93, attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/lightning_indexer/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/lightning_indexer/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/lightning-indexer.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/lightning_indexer/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/lightning_indexer/)。

# LightningIndexer（稀疏注意力索引选择器）

CANN ops-transformer 的 `LightningIndexer` 是稀疏注意力（SparseFlashAttention）的**索引选择器**：根据 Q_index 和 K_index 计算重要性分数，经 ReLU 激活和 weights 缩放后，topK 选出每个 token 对应的最重要 KV 位置（sparse_indices），供下游稀疏注意力算子使用。来源见 `ops-transformer-lightning-indexer`。

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

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, gqa
- 实践案例: [Elementwise 算子优化](../reference/op-class-elementwise.md) — 主题相关：共享 relu
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention, gqa, sparse_attention
- 相关主题: [FlashAttention 变体的内存访问优化](../runbooks/algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 attention, gqa
- 相关主题: [Attention 硬件适配优化聚合](../runbooks/portability/traj-attention-hardware_adaptation.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, sparse_attention, sparse_kv_gather, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, sparse_attention, sparse_kv_gather, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, sparse_attention, sparse_kv_gather, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, topk, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, topk, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — lightning indexer 的量化变体
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, sparse_attention, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, page_attention, sparse_attention, sparse_kv_gather, unified_buffer

<!-- okf:related:end -->
