---
id: kernel-moe-finalize-routing-v2
title: MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU
type: kernel
tags: [moe, finalize_routing, gather, reduce, elementwise, aiv_only]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR, Ascend310P1, Kirin9030, KirinX90]
kernel_types: [moe]
operators: [moe_finalize_routing_v2]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-finalize-routing-v2
related: []
sources: [ops-transformer-moe-finalize-routing-v2]
performance_claims: []
---

# MoeFinalizeRoutingV2 (MoE FFN 输出合并)

CANN ops-transformer 中的 `MoeFinalizeRoutingV2` 算子在 MoE 计算中最后合并 FFN 输出结果。纯 AIV-only Vector 算子（无 matmul），核心操作是 gather + scale + bias + reduce sum。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| expanded_x | 必选 | MoE FFN 输出 | FP16/BF16/FLOAT |
| expanded_row_idx | 必选 | 行索引映射 | INT32 |
| x1 | 可选 | 残差连接 1 | FP16/BF16/FLOAT |
| x2 | 可选 | 残差连接 2 | FP16/BF16/FLOAT |
| bias | 可选 | 专家偏置 | FP16/BF16/FLOAT |
| scales | 可选 | 缩放系数 | FP16/BF16/FLOAT |
| expert_idx | 可选 | 专家索引 | INT32 |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| y | 合并输出 | FP16/BF16/FLOAT |

## 2. 核心公式
```
out(i,j) = x1(i,j) + x2(i,j) + Σ_k (scales(i,k) * (expandedX[expandedRowIdx(i+k*num_rows), j] + bias(expertIdx[i,k], j)))
```

## 3. 模板 dispatch

~50 个 tiling key，按 dtype × 算法策略（CutK/Db/CuthK2/CuthK4/Cuth/AllBias）× 是否有 bias 区分。`KERNEL_TYPE_AIV_ONLY`。

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-finalize-routing-v2]]。