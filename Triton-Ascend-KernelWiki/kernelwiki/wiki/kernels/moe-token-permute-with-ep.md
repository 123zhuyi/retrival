---
id: kernel-moe-token-permute-with-ep
title: MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU
type: kernel
tags: [moe, token_permute, ep, expert_parallel, sort, gather, index_copy, aiv_only, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [moe]
operators: [moe_token_permute_with_ep]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-permute-with-ep
related: [kernel-moe-token-permute, kernel-moe-init-routing-quant, kernel-moe-finalize-routing]
sources: [ops-transformer-moe-token-permute-with-ep]
performance_claims: []
---

# MoeTokenPermuteWithEp（MoE token 重排 + Expert Parallel）

CANN ops-transformer 的 `MoeTokenPermuteWithEp` 是 [[kernel-moe-token-permute]] 的 **EP（Expert Parallel）扩展版**：新增 probs 输入/输出 + range 范围切片（EP 切分）（README L16）。3 阶段：argSort -> argSort -> index_copy（tokens + probs 双输出，按 range 切片）。**AIV-only**。

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
2. 与 [[kernel-moe-token-permute]] 区分：EP 版新增 probs 双输出 + range 范围切片。
3. range 空时框架层回退到 token_permute。
4. Triton 中 argSort 需自定义（bitonic）或 host 预排序。
5. SpiltD 变体为 D 维切分优化（BLOCK_D constexpr）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。