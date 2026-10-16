---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU
description: CANN ops-transformer 的 MoeTokenUnpermuteGrad 是 kernel-moe-token-unpermute 的反向传播：计算 permutedTokensGrad（scatter 加权）和 probsGrad（topK 求和）（README L16）。AIV-only。
tags: [moe, token_unpermute, grad, backward, scatter, reduce_sum, aiv_only, fp16, bf16, fp32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, moe_token_unpermute_grad]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-token-unpermute-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_grad/)。

# MoeTokenUnpermuteGrad（MoE token unpermute 反向）

CANN ops-transformer 的 `MoeTokenUnpermuteGrad` 是 `kernel-moe-token-unpermute` 的**反向传播**：计算 `permutedTokensGrad`（scatter 加权）和 `probsGrad`（topK 求和）（README L16）。**AIV-only**。

## 1. 输入输出

### 输入（def.cpp L33-60）
- permuted_tokens（REQUIRED，FP16/BF16/FP32）
- unpermuted_tokens_grad（REQUIRED，FP16/BF16/FP32）
- sorted_indices（REQUIRED，INT32）
- probs（OPTIONAL，FP16/BF16/FP32）

### 输出（def.cpp L61-72）
- permuted_tokens_grad（REQUIRED）
- probs_grad（REQUIRED）

### 属性（def.cpp L73-74）
`padded_mode`(false)/`restore_shape`({1,1})。

### 约束（README L118-129）
topK≤512；950 转调 finalize_routing_v2_grad。

## 2. 核心公式（README L19-45）

```
有 probs:
  probsGrad = Σ_{k}(unpermutedTokensGrad * permutedTokens[sortedIndices])   # topK 求和
  permutedTokensGrad[sortedIndices[k]] = unpermutedTokensGrad * probs[k]    # scatter 加权

无 probs (topK=1):
  permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]            # 纯 scatter
```

## 3. tiling key dispatch（4 key）

| key | padded_mode | haveProbs | 模板 |
|---|---|---|---|
| 0/10 | false/true | false | ProbNone |
| 1/11 | false/true | true | ProbNotNone |

## 4. 架构支持（README L7-12）

仅 Ascend910B1/910_93（A2/A3）。950 转调 finalize_routing_v2_grad。

## 5. 与 unpermute 配对

- 正向 `moe_token_unpermute`：gather + probs mul + reduce_sum
- 反向 `moe_token_unpermute_grad`：scatter 加权 + probsGrad reduce_sum

## 6. Agent actions

1. 识别为 MoE token unpermute 反向（scatter + 元素乘 + reduce_sum，AIV-only）。
2. 与 `kernel-moe-token-unpermute` 正向配对。
3. 有 probs 时输出双梯度（permutedTokensGrad + probsGrad），无 probs 时仅 permutedTokensGrad。
4. 950 转调 finalize_routing_v2_grad。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Gather/Scatter 算子优化](../reference/op-class-gather_scatter.md) — 主题相关：共享 moe, scatter
- 实践案例: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../runbooks/algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 实践案例: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../runbooks/algorithm-numerics/fp32-reduction-accumulator.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../runbooks/meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../runbooks/meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../runbooks/algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Activation Scheduling 优化聚合](../runbooks/compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 fp32
- 相关主题: [Scatter Weight Gradient 向量化计算优化](../runbooks/compute-pipeline/traj-gather_scatter-compute.md) — 同属优化知识，主题相关
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](../runbooks/compute-pipeline/traj-gather_scatter-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Gather/Scatter 内存访问优化聚合](../runbooks/memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 moe
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, backward, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, backward, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, token_unpermute, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 同一 MoeTokenUnpermute 算子的前向与反向实现
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
