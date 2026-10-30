---
id: kernel-moe-init-routing-v2-grad
title: MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU
type: kernel
tags: [moe, init_routing, backward, gradient, gather, reduce, aiv_only]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [moe]
operators: [moe_init_routing_v2_grad]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-init-routing-v2-grad
related: []
sources: [ops-transformer-moe-init-routing-v2-grad]
performance_claims: []
---

# MoeInitRoutingV2Grad (MoE init routing 反向)

CANN ops-transformer 中的 `MoeInitRoutingV2Grad` 是 `MoeInitRoutingV2` 的反向传播算子，完成 tokens 的加权求和（反向 gather + reduce sum）。纯 AIV-only Vector 算子（无 matmul）。

## 1. 输入输出

### 输入
| 输入 | 必选 | 含义 | dtype |
|------|------|------|-------|
| grad_expanded_x | 是 | Routing 后的目标张量梯度 | FP16/BF16/FLOAT |
| expanded_row_idx | 是 | 行索引映射 | INT32 |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| grad_x | 输入梯度 | FP16/BF16/FLOAT |

### 属性
`top_k`（必需）、`drop_pad_mode`(0)、`active_num`(0)

## 2. 核心公式
```
gradX_i = Σ_{t=0}^{topK} gradExpandedX[expandedRowIdx[i*topK + t]]
```

按 expandedRowIdx 反向 gather gradExpandedX，跨 topK 维 reduce sum。

## 3. 模板 dispatch

`KERNEL_TYPE_AIV_ONLY`。9 tiling key：3 模式（Dropless 1000-1002 / Activate 1010-1012 / PositionPad0 1100-1102）× 3 dtype。3 模板类：`MoeInitRoutingV2GradDroplessCompute` / `ActivateCompute` / `PositionPad0Compute`。arch35 regbase 有 3 套 TilingData（SplitH 200xxx / FullLoad 300xxx / 通用 400xxx）。

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-init-routing-v2-grad]]。