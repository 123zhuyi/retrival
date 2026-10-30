---
id: kernel-moe-token-permute
title: MoeTokenPermute (MoE token permute by indices) on Ascend NPU
type: kernel
tags: [moe, token_permute, sort, gather, index_copy, aiv_only, fp16, bf16, fp32, int8]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [moe]
operators: [moe_token_permute]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-permute
related: [kernel-moe-init-routing-quant, kernel-moe-finalize-routing]
sources: [ops-transformer-moe-token-permute]
performance_claims: []
---

# MoeTokenPermute（MoE token 重排）

CANN ops-transformer 的 `MoeTokenPermute` 是 **MoE token 重排**算子：根据 indices 将 tokens 广播并排序（README L14-16）。3 阶段：argSort -> argSort -> index_copy。**AIV-only**。与 [[kernel-moe-init-routing-quant]] 配对（init routing 排序 expert，token_permute 排序 token）。

## 1. 输入输出

### 输入（def.cpp L23-36）
- tokens（REQUIRED，FP16/BF16/FP32/INT8）
- indices（REQUIRED，INT32/INT64）

### 输出（def.cpp L37-48）
- permute_tokens（REQUIRED，同 tokens dtype）
- sorted_indices（REQUIRED，INT32）

### 属性（def.cpp L49-50）
`num_out_tokens`(0)/`padded_mode`(false)。

### 约束（README L100-113）
indices 元素 < 16777215；不支持 paddedMode=True；topK≤512；950 转调 moe_init_routing_v2。

## 2. 核心公式（README L18-40，paddedMode=false）

```
1. sortedIndicesFirst = argSort(indices)
2. sortedIndicesOut = argSort(sortedIndicesFirst)
3. permuteTokens[sortedIndicesFirst[i]] = tokens[i // topK]
```

## 3. tiling key dispatch（8 key）

| key | 排序 | index_copy | padded |
|---|---|---|---|
| 1 | OneCore | Normal | false |
| 2 | MultiCore | Normal | false |
| 3 | OneCore | SpiltD | false |
| 4 | MultiCore | SpiltD | false |
| 5-8 | 同上 | true | true |

## 4. 架构支持（README L7-12）

仅 Ascend910B1/910_93（A2/A3）。950 在框架层转调 `aclnnMoeInitRoutingV2`。

## 5. Agent actions

1. 识别为 MoE token 重排（argSort + index_copy，AIV-only）。
2. 与 [[kernel-moe-init-routing-quant]] 配对：init routing 排序 expert，token_permute 排序 token。
3. Triton 中 argSort 需自定义（bitonic）或 host 预排序。
4. SpiltD 变体为 D 维切分优化（BLOCK_D constexpr）。
5. 950 转调 V2，不走本 kernel。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。