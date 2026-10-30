---
id: kernel-moe-compute-expert-tokens
title: MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU
type: kernel
tags: [moe, compute_expert_tokens, binary_search, aiv_only, int32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR, Kirin9030, KirinX90]
kernel_types: [moe]
operators: [moe_compute_expert_tokens]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-compute-expert-tokens
related: [kernel-moe-init-routing-quant, kernel-moe-gating-top-k-softmax]
sources: [ops-transformer-moe-compute-expert-tokens]
performance_claims: []
---

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
2. 前提：sortedExperts 已排序（来自 [[kernel-moe-init-routing-quant]] 排序输出）。
3. 3 模板按数据量选择（SS/M/L）。
4. Triton 中二分查找需用 while 循环或自定义并行实现。
5. 是 MoE pipeline 的辅助算子，统计每个专家处理的 token 数量。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。