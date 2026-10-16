---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU
description: CANN ops-transformer 的 NsaSelectedAttention 是 NSA（Native Sparse Attention）训练前向的第二阶段：根据 topKIndices（来自 nsacompressattentioninfer）gather 选中的 KV 块，执行标准 attention 计算，输出 attentionOut + softmaxMax/Sum（供反向梯度使用）。来源见 ops-transformer-nsa-selected-...
tags: [nsa, native_sparse_attention, selected_attention, topk_gather, online_softmax, gqa, training, fp32_accumulation, tail_mask, unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2, ascend910b1, attention, nsa_selected_attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_selected_attention/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_selected_attention/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/nsa-selected-attention.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_selected_attention/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_selected_attention/)。

# NsaSelectedAttention（NSA 选择注意力训练前向）

CANN ops-transformer 的 `NsaSelectedAttention` 是 NSA（Native Sparse Attention）训练前向的第二阶段：根据 `topKIndices`（来自 `nsa_compress_attention_infer`）gather 选中的 KV 块，执行标准 attention 计算，输出 attentionOut + softmaxMax/Sum（供反向梯度使用）。来源见 `ops-transformer-nsa-selected-attention`。

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

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, gqa
- 实践案例: [禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）](../runbooks/portability/no-pytorch-in-forward.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention, gqa, online_softmax
- 相关主题: [FlashAttention 变体的内存访问优化](../runbooks/algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 attention, gqa, online_softmax
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, training, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, native_sparse_attention, nsa, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — NSA 压缩/选择注意力算子家族
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910b1, attention, fp32_accumulation, nsa, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910b1, attention, fp32_accumulation, nsa, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer

<!-- okf:related:end -->
