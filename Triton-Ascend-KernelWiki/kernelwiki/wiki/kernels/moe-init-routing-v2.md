---
id: kernel-moe-init-routing-v2
title: MoeInitRoutingV2 (MoE routing v2) on Ascend NPU
type: kernel
tags: [moe, routing, sort, gather, expert, capacity, drop_pad, mixture_of_experts]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR, Ascend950DT, Ascend310P1, Ascend310P3, Ascend310P5, Ascend310P7]
kernel_types: [moe]
operators: [moe_init_routing_v2]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-init-routing-v2
related: [kernel-moe-init-routing]
sources: [ops-transformer-moe-init-routing-v2]
performance_claims: []
---

# MoeInitRoutingV2（MoE 路由 v2）

CANN ops-transformer 的 `MoeInitRoutingV2` 是 MoE 路由 v2 版本。与 v1 相比：无 rowIdx 输入，新增 expert capacity 控制、drop/pad 模式、expert token 计数输出。来源见 [[ops-transformer-moe-init-routing-v2]]。

## 1. 输入输出

### 输入
- x（FP16/BF16/FP32）、expertIdx（INT32）

### 输出
- expandedX、expandedRowIdx（INT32）
- expert_tokens_count_or_cumsum（INT32，可选）
- expert_tokens_before_capacity（INT32，可选）

### 属性
- active_num（默认0）、expert_capacity（默认0）、expert_num（默认0）、drop_pad_mode（默认0）、expert_tokens_count_or_cumsum_flag（默认0）、expert_tokens_before_capacity_flag（默认false）

## 2. 公式

`sort(expertIdx)` → `srcToDstWithCapacity(expertIdx, sortedRowIdx, expert_capacity)` → `expandedRowIdx / expandedX` → `expertTokenOut`。

## 3. 架构支持

A2 √、A3 √、950 ✓、**310P √**（v1 无 310P）。纯 Vector 核（MIX_AIV_1_0）。

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。