---
id: kernel-moe-token-permute-with-routing-map
title: MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU
type: kernel
tags: [moe_token_permute_with_routing_map, moe, token_permute, routing_map, masked_select, argsort, gather]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, vector_core]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93, Ascend950PR, Ascend950DT]
kernel_types: [moe]
operators: [moe_token_permute_with_routing_map]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-permute-with-routing-map
related: [kernel-moe-token-permute-grad]
sources: [ops-transformer-moe-token-permute-with-routing-map]
performance_claims: []
---

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

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、多核归并排序映射、arch35 SIMD/SIMT gather 精度需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-token-permute-with-routing-map]]。