---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU
description: CANN ops-transformer 的 MoeComputeExpertTokens 是 MoE 辅助算子：通过二分查找在已排序的 sortedExperts 中找每个专家处理的最后一行位置，输出累积 token 数（README L18）。AIV-only。
tags: [moe, compute_expert_tokens, binary_search, aiv_only, int32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, ascend950pr, kirin9030, kirinx90, moe_compute_expert_tokens]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_compute_expert_tokens/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_compute_expert_tokens/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-compute-expert-tokens.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_compute_expert_tokens/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_compute_expert_tokens/)。

# MoeComputeExpertTokens（MoE 专家 token 数统计）

CANN ops-transformer 的 `MoeComputeExpertTokens` 是 **MoE 辅助算子**：通过二分查找在已排序的 sortedExperts 中找每个专家处理的最后一行位置，输出累积 token 数（README L18）。**AIV-only**。

## 1. 输入输出

### 输入（def.cpp L23-28）
- sorted_experts（REQUIRED，INT32）：已排序的专家索引（来自 init_routing 排序输出）。

### 输出（def.cpp L29-33）
- total_rows_before_expert（REQUIRED，INT32）：每个专家的累积 token 数。

### 属性（def.cpp L34）
`num_experts`（INT，总专家数）。

### 约束（README L62-66）
sortedExperts shape < 2^24；numExperts >0 且 ≤2048。

## 2. 核心公式（README L21-23）

```
out[i] = BinarySearch(sortedExperts, numExperts)
```

在已排序的 sortedExperts 中，对每个专家 i 二分查找其最后出现的行位置。

## 3. tiling key dispatch（3 key）

| key | 模板 | 场景 |
|---|---|---|
| 1001 | Int32SS | 小数据量 |
| 1002 | Int32M | 中数据量 |
| 1003 | Int32L | 大数据量 |

## 4. 架构支持（README L7-14）

A2/A3 + 950 + KirinX90/Kirin9030（全架构统一 config）。

## 5. Agent actions

1. 识别为 MoE 专家 token 数统计（二分查找，AIV-only）。
2. 前提：sortedExperts 已排序（来自 `kernel-moe-init-routing-quant` 排序输出）。
3. 3 模板按数据量选择（SS/M/L）。
4. Triton 中二分查找需用 while 循环或自定义并行实现。
5. 是 MoE pipeline 的辅助算子，统计每个专家处理的 token 数量。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。

<!-- okf:related:start -->

# 相关

- 实践案例: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](../runbooks/compute-pipeline/grid-by-output-count.md) — 同属优化知识，主题相关
- 实践案例: [整数比较/加法必须转 float32，避免 Vector 标量降级](../runbooks/compute-pipeline/int-compare-cast-float32.md) — 主题相关：共享 int32
- 实践案例: [kernel 内索引必须转 int32，避免 int64 降级](../runbooks/compute-pipeline/int32-index-cast.md) — 同属优化知识，主题相关
- 实践案例: [Work-Stealing 动态负载均衡（不均匀任务分发）](../runbooks/compute-pipeline/work-stealing-dynamic-load-balance.md) — 主题相关：共享 moe
- 实践案例: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../runbooks/memory-sram/gather-random-address-slow.md) — 主题相关：共享 binary_search
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, kirin9030, kirinx90, unified_buffer

<!-- okf:related:end -->
