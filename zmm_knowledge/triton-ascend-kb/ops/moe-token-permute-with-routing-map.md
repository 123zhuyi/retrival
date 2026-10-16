---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU
description: CANN ops-transformer 中的 MoeTokenPermuteWithRoutingMap 算子是用 routingMap（token 到 expert 映射）驱动的 MoE token 置换算子。核心是 maskedSelect + argSort + indexcopy/gather，支持两种模式（dropAndPad false/true）。三架构（A2/A3 + 950 arch35，arch35 含 SIMD/SIMT 二维 gather 专属...
tags: [moe_token_permute_with_routing_map, moe, token_permute, routing_map, masked_select, argsort, gather, fp32_accumulation, unified_buffer, vector_core, ascend910b1, ascend910b2, ascend910_93, ascend950pr, ascend950dt]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_routing_map/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_routing_map/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-token-permute-with-routing-map.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_routing_map/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_with_routing_map/)。

# MoeTokenPermuteWithRoutingMap (routingMap-driven token permute)

CANN ops-transformer 中的 `MoeTokenPermuteWithRoutingMap` 算子是用 routingMap（token 到 expert 映射）驱动的 MoE token 置换算子。核心是 maskedSelect + argSort + index_copy/gather，支持两种模式（dropAndPad false/true）。三架构（A2/A3 + 950 arch35，arch35 含 SIMD/SIMT 二维 gather 专属模板）。

## 1. 输入输出

### 输入（按 `moe_token_permute_with_routing_map_def.cpp`）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| tokens | 必选 | 输入 tokens [tokens_num, H] | BF16/FP16/FP32 |
| routing_map | 必选 | token 到 expert 映射 [tokens_num, expert_num] | BOOL/INT8 |
| probs | 可选 | 概率 [tokens_num, expert_num] | BF16/FP16/FP32 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| permute_tokens | 置换后的 tokens [num_out_tokens, H] | BF16/FP16/FP32 |
| permute_probs | 置换后的 probs [num_out_tokens] | BF16/FP16/FP32 |
| sorted_indices | 排序索引 [num_out_tokens] | INT32 |

### 属性

| 属性 | 含义 | 说明 |
|------|------|------|
| num_out_tokens | 有效输出 token 数 | REQUIRED，用于计算 topK/capacity |
| drop_and_pad | 是否开启 dropAndPad 模式 | OPTIONAL，默认 false |

### 约束

- tokens_num 和 expert_num < 16777215
- dropAndPad=false 时 routingMap 每行 1/true 个数固定且 < 512

## 2. 核心计算公式

### dropAndPad=false（topK 模式）

```
expertIndex = arange(tokens_num).expand(expert_num, -1)
sortedIndicesFirst = expertIndex.maskedSelect(routingMap.T)
sortedIndicesOut = argSort(sortedIndicesFirst)
topK = numOutTokens // tokens_num
permuteTokens[sortedIndicesOut[i]] = tokens[i // topK]
permuteProbsOut = probs.T.maskedSelect(routingMap.T)
```

### dropAndPad=true（capacity 模式）

```
capacity = numOutTokens // expert_num
sortedIndicesOut = argsort(routingMap.T, dim=-1)[:, :capacity]
permutedTokensOut = tokens.index_select(0, sortedIndicesOut)
# probs: 按 (expert_num, capacity) 二维索引 gather
```

## 3. 计算流程

```
1. KernelMaskedSelectV3: routingMap.T -> sortedIndicesFirst
2. MoeSort（OneCore/MultiCore）: sortedIndicesFirst -> sortedIndicesOut
   - 多核：分核排序 + moe_mrgsort 归并
3. MoeindexCopyOp: tokens.index_select(sortedIndicesOut) -> permuteTokens
4. MoePermuteProb: probs 置换 -> permuteProbsOut
5. arch35: Gatherv2Simd/SimtTwoDim（二维 gather 优化）
```

## 4. 模板与 dispatch

| TilingKey | 模式 | 说明 |
|---|---|---|
| 1/3/5/7 | 单核排序 | dropAndPad=false，MoeSortOneCore |
| 2/4/6/8 | 多核排序 | dropAndPad=false，MoeSortMultiCore + 归并 |
| 9/19 | 填充模式 | dropAndPad=true，仅 3510/arch35 |

## 5. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 |
|---|---|---|---|
| Atlas A2 训练/推理 | arch32 | `ascend910b` L80 | `moe_token_permute_with_routing_map.cpp` |
| Atlas A3 训练/推理 | ascend910_93 | `ascend910_93` L81 | 同上 |
| Ascend 950PR/950DT | arch35 | `ascend950` L88 | 同上（regbase + SIMD/SIMT gather） |

> 三架构均支持。950 用 arch35 专属 `Gatherv2SimdTwoDim`/`Gatherv2SimtTwoDim`。支持 BF16/FP16/FP32 tokens + BOOL/INT8 routingMap。

## 6. Agent actions

1. 确认目标架构（arch32 / arch35）与模式（dropAndPad false/true）。
2. maskedSelect：`tl.load(tokens + mask_nonzero_indices)`。
3. argSort：`tl.argsort` 或自定义排序（多核需归并）。
4. index_select：`tl.load(tokens + sorted_indices)`。
5. arch35：SIMD/SIMT 二维 gather 优化（不迁移机制，迁移语义）。
6. Triton 实现参考 `artifacts/moe-token-permute-with-routing-map/moe-token-permute-with-routing-map-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、多核归并排序映射、arch35 SIMD/SIMT gather 精度需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-moe-token-permute-with-routing-map`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Gather/Scatter 算子优化](../reference/op-class-gather_scatter.md) — 主题相关：共享 gather, moe
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../runbooks/compute-pipeline/exp_causal_conv1d_update_006.md) — 同属优化知识，主题相关
- 实践案例: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../runbooks/memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 实践案例: [Tiling 预算与布局驱动的块大小选择](../runbooks/memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 实践案例: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../runbooks/meta-skills/pipeline-diagram-driven-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [形状驱动算子决策树](../runbooks/meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, fp32_accumulation, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer, vector_core
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, moe, token_permute, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, token_permute, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, token_permute, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, token_permute, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, fp32_accumulation, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core

<!-- okf:related:end -->
