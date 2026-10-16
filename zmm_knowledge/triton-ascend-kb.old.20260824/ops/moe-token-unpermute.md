---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU
description: CANN ops-transformer 的 MoeTokenUnpermute 是 kernel-moe-token-permute 的逆操作：按 sortedIndices 从 permutedTokens gather，若有 probs 相乘，再按 topK 累加求和还原回原始 token 顺序（README L18）。AIV-only（gather + probs mul + reducesum，无 Cube）。
tags: [moe, token_unpermute, gather, reduce_sum, aiv_only, fp16, bf16, fp32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, moe_token_unpermute]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-token-unpermute.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute/)。

# MoeTokenUnpermute（MoE token 反向重排）

CANN ops-transformer 的 `MoeTokenUnpermute` 是 `kernel-moe-token-permute` 的**逆操作**：按 sortedIndices 从 permutedTokens gather，若有 probs 相乘，再按 topK 累加求和还原回原始 token 顺序（README L18）。**AIV-only**（gather + probs mul + reduce_sum，无 Cube）。

## 1. 输入输出

### 输入（def.cpp L23-58）
- permuted_tokens（REQUIRED，FP16/BF16/FP32）
- sorted_indices（REQUIRED，INT32）
- probs（OPTIONAL，FP16/BF16/FP32）

### 输出（def.cpp L63-73）
- unpermuted_tokens（REQUIRED，同 permuted_tokens dtype）

### 属性（def.cpp L60-61）
`padded_mode`(false)/`restore_shape`(OPTIONAL ListInt)。

### 约束（README L106-119）
topK≤512；950 转调 finalize_routing_v2；310P 仅 FP16/FP32，hiddensize 128 倍数≤10240。

## 2. 核心公式（README L21-47）

```
有 probs:
  T[k] = permutedTokens[sortedIndices[k]]       # gather
  T[k] = T[k] * probs[k]                          # probs 乘
  O[i] = Σ_{k=i*topK}^{(i+1)*topK-1} T[k]        # topK 累加还原

无 probs (topK=1):
  O[i] = permutedTokens[sortedIndices[i]]         # 纯 gather
```

## 3. tiling key dispatch（~16 key）

dtype(BF16/FP16/FP32) × haveProbs(false/true) × mix mode（probs dtype ≠ token dtype，key 11/13/17/21/25/27）。

## 4. 架构支持（README L7-14）

A2/A3(BF16/FP16/FP32) + 310P/Kirin(仅 FP16/FP32)。950 转调 finalize_routing_v2。

## 5. 与 token_permute 配对

- `moe_token_permute`：tokens -> permuteTokens（按 indices 排序展开）
- `moe_token_unpermute`：permutedTokens -> unpermutedTokens（按 sortedIndices gather + probs 加权 + topK 累加还原）

## 6. Agent actions

1. 识别为 MoE token 反向重排（gather + probs mul + reduce_sum，AIV-only）。
2. 与 `kernel-moe-token-permute` 配对：permute 排序展开，unpermute gather+累加还原。
3. mix mode 支持 probs dtype 与 token dtype 不同（如 FP16 token + FP32 probs）。
4. 950 转调 finalize_routing_v2。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。

<!-- okf:related:start -->

# 相关

- 实践案例: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../runbooks/algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 实践案例: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../runbooks/algorithm-numerics/fp32-reduction-accumulator.md) — 同属优化知识，主题相关
- 实践案例: [跨 tile 归约必须显式两阶段 reduce-then-apply](../runbooks/algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../runbooks/meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../runbooks/algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 bf16, fp16
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 同一 MoeTokenUnpermute 算子的前向与反向实现
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, token_unpermute, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
