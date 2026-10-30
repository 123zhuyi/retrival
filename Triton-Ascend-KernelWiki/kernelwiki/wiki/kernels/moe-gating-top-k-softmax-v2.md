---
id: kernel-moe-gating-top-k-softmax-v2
title: MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU
type: kernel
tags: [moe, gating, topk, softmax, aiv_only]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [moe]
operators: [moe_gating_top_k_softmax_v2]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-gating-top-k-softmax-v2
related: []
sources: [ops-transformer-moe-gating-top-k-softmax-v2]
performance_claims: []
---

# MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax)

CANN ops-transformer 中的 `MoeGatingTopKSoftmaxV2` 是 MoE gating 的 topk + softmax 算子，按 renorm 标记区分 softmax 与 topk 的先后顺序。纯 AIV-only Vector 算子（无 matmul）。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| x | 必选 | gating 输入 | FP16/FLOAT/BF16 |
| finished | 可选 | 行完成标记（True 则 expert_idx=num_expert） | BOOL |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| y | softmax 的 topk 结果 | FP16/FLOAT/BF16 |
| expert_idx | topk 的 indices（专家序号） | INT32 |
| softmax_result | softmax 结果（可选） | FLOAT |

### 属性
`k`（必需）、`renorm`(0)、`output_softmax_result_flag`(false)

## 2. 核心公式
- **renorm=0**：`softmaxResult = softmax(x, axis=-1)`；`yOut, expertIdxOut = topk(softmaxResult, k)`
- **renorm=1**：`topkOut, expertIdxOut = topk(x, k)`；`yOut = softmax(topkOut, axis=-1)`
- finished=True 时 expertIdxOut = num_expert

## 3. 模板 dispatch

`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。~36 tiling key，4 模板策略：
- **EKFullLoad**（101xxx）：expert+K 全加载
- **KFullLoad**（102xxx，renorm=0）：K 全加载
- **KRenorm**（102xxx，renorm=1）：K renorm
- **Perf**（103xxx）：高性能，按 colRange 分 3 档（<8 / 8-64 / >64）

## 4. 约束
- k <= 1024
- renorm 只支持 0/1
- x 和 finished 每一维 <= INT32_MAX

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-gating-top-k-softmax-v2]]。