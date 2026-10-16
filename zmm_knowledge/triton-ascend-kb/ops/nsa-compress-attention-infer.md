---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU
description: CANN ops-transformer 的 NsaCompressAttentionInfer 是 NSA（Native Sparse Attention）推理流水线的第一阶段：在压缩后的 KV 上执行注意力计算，产出重要性分数（滑动窗口卷积），最终选择 topK 个最重要 KV 块索引，供后续 nsaselectedattentioninfer 使用。来源见 ops-transformer-nsa-compress-attention-infer。
tags: [nsa, native_sparse_attention, compress_attention, sliding_window, importance_score, topk, paged_attention, gqa, cube_vector_pipe, fp32_accumulation, tail_mask, unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2, ascend910b1, ascend910_93, attention, nsa_compress_attention_infer]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_attention_infer/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_attention_infer/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/nsa-compress-attention-infer.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_attention_infer/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_attention_infer/)。

# NsaCompressAttentionInfer（NSA 压缩注意力推理）

CANN ops-transformer 的 `NsaCompressAttentionInfer` 是 NSA（Native Sparse Attention）推理流水线的第一阶段：在压缩后的 KV 上执行注意力计算，产出重要性分数（滑动窗口卷积），最终选择 topK 个最重要 KV 块索引，供后续 `nsa_selected_attention_infer` 使用。来源见 `ops-transformer-nsa-compress-attention-infer`。

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

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, gqa
- 实践案例: [流程优化: 标准 score-归一化-加权三步流程](../runbooks/compute-pipeline/exp_fused_attention_001.md) — 主题相关：共享 attention
- 实践案例: [Triton-Ascend 移位 load 降级契约（大位移 ~25x 标量化，2D tile ±1 不合并）](../runbooks/compute-pipeline/triton-ascend-shifted-load-degradation.md) — 主题相关：共享 sliding_window
- 实践案例: [Memory Access Optimization: paged kv cache with gqa head tiling](../runbooks/memory-sram/exp_paged_gqa_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: score-tile reuse with online softmax](../runbooks/memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention
- 实践案例: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../runbooks/memory-sram/overlapping-t-tiling.md) — 主题相关：共享 sliding_window
- 实践案例: [寄存器移位复用滑动窗口，消除重复 HBM 读](../runbooks/memory-sram/register-shift-sliding-window.md) — 主题相关：共享 sliding_window
- 实践案例: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../runbooks/portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, paged_attention, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, paged_attention, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, topk, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — NSA 压缩注意力的训练/推理变体
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, nsa, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, nsa, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, native_sparse_attention, nsa, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, paged_attention, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, topk, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer

<!-- okf:related:end -->
