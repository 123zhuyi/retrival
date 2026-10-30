---
id: kernel-moe-init-routing
title: MoeInitRouting (MoE routing) on Ascend NPU
type: kernel
tags: [moe, routing, sort, gather, expert, mixture_of_experts]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR, Ascend950DT]
kernel_types: [moe]
operators: [moe_init_routing]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-init-routing
related: []
sources: [ops-transformer-moe-init-routing]
performance_claims: []
---

# MoeInitRouting（MoE 路由）

CANN ops-transformer 的 `MoeInitRouting` 是 MoE（Mixture of Experts）路由处理的第一阶段：根据 `MoeGatingTopKSoftmax` 的输出（expertIdx），对 token 特征按 expert 排序和扩展，输出 expandedX（按 expert 排列的 token 特征）、expandedRowIdx（映射关系）、expandedExpertIdx（排序后的 expert 索引）。来源见 [[ops-transformer-moe-init-routing]]。

## 1. 输入输出

### 输入（def.cpp L23-40）

| 输入 | 必选 | dtype | 含义 |
|---|---|---|---|
| x | 必选 | FP16/BF16/FP32 | token 特征输入 |
| rowIdx | 必选 | INT32 | 每个位置对应的原始行位置 |
| expertIdx | 必选 | INT32 | MoeGatingTopKSoftmax 的输出，每行对应的 K 个专家 |

### 输出（def.cpp L41-55）

| 输出 | dtype | 含义 |
|---|---|---|
| expandedX | FP16/BF16/FP32 | 按 expert 排序扩展后的特征 |
| expandedRowIdx | INT32 | expandedX 和 x 的映射关系 |
| expandedExpertIdx | INT32 | 排序后的 expertIdx |

### 属性（def.cpp L56）
- activeNum（INT64，必选）：总的最大处理 row 数，expandedXOut 只有这么多行有效。

## 2. 核心公式

`expandedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)`
`expandedRowIdx[sortedRowIdx[i]] = i`
`expandedX[i] = x[sortedRowIdx[i] % numRows]`

## 3. 架构支持

A2（Ascend910B1）√、A3（Ascend910_93）√、950（Ascend950PR/Ascend950DT）√。纯 Vector 核（MIX_AIV_1_0），无需 Cube。

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。