---
id: kernel-moe-finalize-routing-v2-grad
title: MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU
type: kernel
tags: [moe_finalize_routing_v2_grad, moe, routing, backward, scatter_add, elementwise, reduce]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [moe]
operators: [moe_finalize_routing_v2_grad]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-finalize-routing-v2-grad
related: []
sources: [ops-transformer-moe-finalize-routing-v2-grad]
performance_claims: []
---

# MoeFinalizeRoutingV2Grad (MoE routing finalize backward)

CANN ops-transformer 中的 `MoeFinalizeRoutingV2Grad` 算子是 `MoeFinalizeRoutingV2` 的反向传播算子。核心是 expandedRowIdx 驱动的 scatter-add（gradY 按索引映射到 gradExpandedXOut）+ 可选 scale 梯度计算。三架构（A2/A3/950），纯 Vector 计算，无 Cube。

## 1. 输入输出

### 输入（按 `moe_finalize_routing_v2_grad_def.cpp`）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| gradY | 必选 | 下游梯度 [R, H] | FP16/BF16/FP32 |
| expandedRowIdx | 必选 | 专家序排序索引 [R*K] | INT32 |
| expandedX | 可选 | 扩展后的特征 [R*K, H] | FP16/BF16/FP32 |
| scales | 可选 | 缩放因子 [R, K] | FP16/BF16/FP32 |
| expertIdx | 可选 | 专家索引 [R, K] | INT32 |
| bias | 可选 | 偏移 [E, H] | FP16/BF16/FP32 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| gradExpandedXOut | expandedX 的梯度 | FP16/BF16/FP32 |
| gradScalesOut | scales 的梯度（有 scale 时） | FP16/BF16/FP32 |

### 属性

`dropPadMode`（场景选择）、`activeNum`（最大输出行数）、`expertNum`（专家数）、`expertCapacity`（专家容量）。

## 2. 核心计算公式

```
场景 1（无 scale）：
  gradExpandedXOut[expandedRowIdx[i]][j] += gradY[i/K][j]

场景 2（有 scale）：
  gradExpandedXOut[expandedRowIdx[i]][j] += gradY[i/K][j] * scales[i]
  gradScalesOut[i] = sum(expandedX[expandedRowIdx[i]][j] * gradY[i/K][j])

场景 3（有 scale + bias）：
  gradExpandedXOut[expandedRowIdx[i]][j] += gradY[i/K][j] * scales[i]
  gradScalesOut[i] = sum((expandedX[expandedRowIdx[i]][j] + bias[expertIdx[i]][j]) * gradY[i/K][j])
```

## 3. 计算流程

```
for each (i, j) in (R*K) × H:
    idx = expandedRowIdx[i]
    gradExpandedX[idx][j] += gradY[i/K][j] * scales[i]    # scatter-add
    if with_scale:
        gradScales[i] += (expandedX[idx][j] | +bias) * gradY[i/K][j]  # reduce
```

## 4. 模板与 dispatch

| TilingKey | 模板 | scale | 切 H | 文件 |
|---|---|---|---|---|
| 10001 | `MoeFinalizeRoutingV2GradWithoutScaleNotCutH` | 无 | 不切 | `without_scale_not_cut_h.h` |
| 10002 | `MoeFinalizeRoutingV2GradWithoutScaleCutH` | 无 | 切 | `without_scale_cut_h.h` |
| 20001 | `MoeFinalizeRoutingV2GradWithScaleNotCutH` | 有 | 不切 | `with_scale_not_cut_h.h` |
| 20002 | `MoeFinalizeRoutingV2GradWithScaleCutH` | 有 | 切 | `with_scale_cut_h.h` |

## 5. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 |
|---|---|---|---|
| Atlas A2 训练/推理 | arch32 | `ascend910b` L104 | `moe_finalize_routing_v2_grad.cpp` |
| Atlas A3 训练/推理 | ascend910_93 | `ascend910_93` L105 | 同上 |
| Ascend 950PR/950DT | arch35 | `ascend950` L149 | `moe_finalize_routing_v2_grad_apt.cpp` |

纯 Vector 计算（`KERNEL_TYPE_MIX_AIV_1_0`），无 Cube。支持 FP16/BF16/FP32。

## 6. Agent actions

1. 确认目标架构（arch32 / arch35）与场景（无 scale / 有 scale / 有 scale+bias）。
2. 核间：expandedRowIdx 驱动的 scatter-add → `tl.atomic_add`。
3. scale 梯度：`tl.sum(expandedX[idx] * gradY)` reduce。
4. Triton 实现参考 `artifacts/moe-finalize-routing-v2-grad/moe-finalize-routing-v2-grad-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、atomic_add 多核精度边界需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-finalize-routing-v2-grad]]。