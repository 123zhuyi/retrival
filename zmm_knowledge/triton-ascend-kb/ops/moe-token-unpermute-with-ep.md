---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU
description: CANN ops-transformer 中的 MoeTokenUnpermuteWithEp 是 MoE token 反重排算子，根据 sortedIndices 从 permutedTokens 取数据（与 probs 相乘）并合并累加。纯 AIV-only Vector 算子（无 matmul），核心操作是 range 过滤 + gather(indexSelect) + 可选乘 probs + 跨 topK reduce sum。支持 A2 + Kirin（不支...
tags: [moe, token_unpermute, ep, gather, reduce, aiv_only, fp32_accumulation, unified_buffer, register_file, vector_core, ascend910b1, ascend910_93, kirin9030, kirinx90, moe_token_unpermute_with_ep]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_with_ep/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_with_ep/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-token-unpermute-with-ep.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_with_ep/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_unpermute_with_ep/)。

# MoeTokenUnpermuteWithEp (MoE token unpermute with EP)

CANN ops-transformer 中的 `MoeTokenUnpermuteWithEp` 是 MoE token 反重排算子，根据 sortedIndices 从 permutedTokens 取数据（与 probs 相乘）并合并累加。纯 AIV-only Vector 算子（无 matmul），核心操作是 range 过滤 + gather(indexSelect) + 可选乘 probs + 跨 topK reduce sum。支持 A2 + Kirin（不支持 950）。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| permuted_tokens | 必选 | 扩展并排序过的 tokens | BF16/FP16/FLOAT |
| sorted_indices | 必选 | 数据在 permutedTokens 中的位置 | INT32 |
| probs | 可选 | 专家概率 | BF16/FP16/FLOAT |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| unpermuted_tokens | 反重排输出 | BF16/FP16/FLOAT |

### 属性
`num_topk`(1)、`range`([0,0]，ep 切分范围)、`padded_mode`(false)、`restore_shape`（仅 nullptr）

## 2. 核心公式
1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `permutedTokens = permutedTokens.indexSelect(0, sortedIndices)` - gather
3. 若 probs 非 None：`permutedTokens_k *= probs_{i,j}` - 乘 probs
4. `out_i = Σ_{k=i*numTopk}^{(i+1)*numTopk-1} permutedTokens_k` - 跨 topK reduce sum

## 3. 模板 dispatch

4 tiling key：0（无 probs）/ 1（float probs）/ 2（half probs）/ 3（bf16 probs）。1 模板类 `KernelMoeTokenUnpermuteWithEp`，通过 `haveProbs` + probs dtype 区分。

## 4. 约束
- topK_num <= 512
- 不支持 paddedMode=True
- rangeOptional 为空时回退到 `MoeTokenUnpermute`
- ascend910b/910_93 + kirinX90/kirin9030（不支持 950）

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-moe-token-unpermute-with-ep`。

<!-- okf:related:start -->

# 相关

- 实践案例: [跨 tile 归约必须显式两阶段 reduce-then-apply](../runbooks/algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../runbooks/compute-pipeline/exp_causal_conv1d_update_006.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, kirin9030, kirinx90, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, reduce, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ep, fp32_accumulation, gather, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ep, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, token_unpermute, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, token_unpermute, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, reduce, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core

<!-- okf:related:end -->
