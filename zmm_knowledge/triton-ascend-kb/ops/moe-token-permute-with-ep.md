---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU
description: CANN ops-transformer 的 MoeTokenPermuteWithEp 是 kernel-moe-token-permute 的 EP（Expert Parallel）扩展版：新增 probs 输入/输出 + range 范围切片（EP 切分）（README L16）。3 阶段：argSort -> argSort -> indexcopy（tokens + probs 双输出，按 range 切片）。AIV-only。
tags: [moe, token_permute, ep, expert_parallel, sort, gather, index_copy, aiv_only, fp16, bf16, fp32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, moe_token_permute_with_ep]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_ep/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_ep/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-token-permute-with-ep.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_ep/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_ep/)。

# MoeTokenPermuteWithEp（MoE token 重排 + Expert Parallel）

CANN ops-transformer 的 `MoeTokenPermuteWithEp` 是 `kernel-moe-token-permute` 的 **EP（Expert Parallel）扩展版**：新增 probs 输入/输出 + range 范围切片（EP 切分）（README L16）。3 阶段：argSort -> argSort -> index_copy（tokens + probs 双输出，按 range 切片）。**AIV-only**。

## 1. 输入输出

### 输入（def.cpp L23-43）
- tokens（REQUIRED，FP16/BF16/FP32）
- indices（REQUIRED，INT32/INT64）
- probs（OPTIONAL，FP16/BF16/FP32）

### 输出（def.cpp L44-61）
- permute_tokens（REQUIRED，同 tokens dtype）
- sorted_indices（REQUIRED，INT32）
- permute_probs（REQUIRED，同 probs dtype）

### 属性（def.cpp L62-64）
`range`({0,0}，EP 切分范围)/`num_out_tokens`(0)/`padded_mode`(false)。

### 约束（README L128-133）
indices 元素 < 16777215；topK≤512；不支持 paddedMode=True；range 空时回退到 moe_token_permute。

## 2. 核心公式（README L22-38，paddedMode=false）

```
1. sortedIndicesFirst = argSort(indices)
2. sortedIndicesOut = argSort(sortedIndicesFirst)
3. 当 range[0] <= sortedIndices[i] < range[1]:
   permuteTokens[sortedIndices[i]-range[0]] = tokens[i//topK]
   permuteProbs[sortedIndices[i]-range[0]] = probs[i]
```

## 3. 与 token_permute 差异

| 差异 | token_permute | token_permute_with_ep |
|---|---|---|
| probs 输入/输出 | 无 | 有 |
| range 属性 | 无 | 有（EP 切分范围） |
| index_copy | 单 tokens | tokens + probs 双输出 |
| 切片逻辑 | numOutTokens | range[0]<=idx<range[1] |
| 回退 | - | range 空时回退到 token_permute |

## 4. tiling key dispatch（8 key，与 token_permute 同构）

| key | 排序 | index_copy | padded |
|---|---|---|---|
| 1 | OneCore | Normal | false |
| 2 | MultiCore | Normal | false |
| 3 | OneCore | SpiltD | false |
| 4 | MultiCore | SpiltD | false |
| 5-8 | 同上 | true | true |

## 5. 架构支持（README L7-12）

仅 Ascend910B1/910_93（A2/A3）。

## 6. Agent actions

1. 识别为 MoE token 重排 EP 版（argSort + 条件 index_copy，AIV-only）。
2. 与 `kernel-moe-token-permute` 区分：EP 版新增 probs 双输出 + range 范围切片。
3. range 空时框架层回退到 token_permute。
4. Triton 中 argSort 需自定义（bitonic）或 host 预排序。
5. SpiltD 变体为 D 维切分优化（BLOCK_D constexpr）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Embedding / 索引拷贝算子优化](../reference/op-class-embedding.md) — 主题相关：共享 gather, index_copy
- 实践案例: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../runbooks/compute-pipeline/exp_causal_conv1d_update_006.md) — 同属优化知识，主题相关
- 实践案例: [Work-Stealing 动态负载均衡（不均匀任务分发）](../runbooks/compute-pipeline/work-stealing-dynamic-load-balance.md) — 主题相关：共享 moe
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../runbooks/memory-sram/gather-random-address-slow.md) — 主题相关：共享 gather
- 实践案例: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../runbooks/memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 实践案例: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../runbooks/memory-sram/parallel-axis-align.md) — 同属优化知识，主题相关
- 实践案例: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../runbooks/portability/exp_gqa_paged_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 aiv_only, bf16, fp16, fp32
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 aiv_only, bf16, fp16, fp32
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 aiv_only, fp16, fp32
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, token_permute, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 同一 MoeTokenPermuteWithEP 算子的前向与反向实现
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, token_permute, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, index_copy, moe, sort, token_permute, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ep, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
