---
id: kernel-moe-finalize-routing
title: MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU
type: kernel
tags: [moe, finalize_routing, elementwise, gather, scale_add, aiv_only, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Kirin9030, KirinX90]
kernel_types: [moe]
operators: [moe_finalize_routing]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-finalize-routing
related: []
sources: [ops-transformer-moe-finalize-routing]
performance_claims: []
---

# MoeFinalizeRouting（MoE FFN 输出路由合并）

CANN ops-transformer 的 `MoeFinalizeRouting` 是 **MoE 最终路由合并**算子：将 MoE FFN 的输出按 routing 索引和 scale 加权合并回原始 token 位置，支持 skip connection（x1+x2）和 expert bias（README L18-19）。**纯 AIV-only elementwise 算子**（gather+scale+add，无 Cube/matmul 参与）。

## 1. 输入输出

### 输入（def.cpp L22-63）

| 输入 | 必选/可选 | 含义 | dtype |
|---|---|---|---|
| expanded_x | 必选 | MoE FFN 输出 | FP16/FP32/BF16 |
| x1 | 必选 | skip connection 1 | FP16/FP32/BF16 |
| x2 | 可选 | skip connection 2 | FP16/FP32/BF16 |
| bias | 必选 | expert 偏置 | FP16/FP32/BF16 |
| scales | 必选 | routing scale | FP16/FP32/BF16 |
| expanded_row_idx | 必选 | expandedX 行索引 | INT32 |
| expanded_expert_idx | 必选 | expert 索引 | INT32 |

### 输出（def.cpp L64-68）
- y（REQUIRED，FP16/FP32/BF16）

### 约束（README L101-103）：无特殊约束。

## 2. 核心公式（README L21-27）

```
out[i, j] = x1[i, j] + x2Optional[i, j]
            + Σ_{k=0}^{K}(scales[i, k]
              * (expandedX[expandedRowIdx[i+k*numRows], j]
                 + bias[expandedExpertIdx[i, k], j]))
```

## 3. tiling key dispatch（19 key, 20000-20018）

dtype(FP32/FP16/BF16) × 模板变体(CutK/Cuth/K2/K4/Db/DbAllBias/AllBias/CuthK2Optimized)。`BUFFER_NUM=1`，`PARALLEL_NUM=2`。

## 4. 架构支持（def.cpp L79-84）

A2/A3(FP16/FP32/BF16) + KirinX90/Kirin9030(仅 FP16/FP32)。不支持 950/310P。

## 5. Agent actions

1. 识别为 MoE 路由合并（纯 gather+scale+add，AIV-only）。
2. Triton 中 `tl.load(expanded_row_idx)` 间接寻址 + `tl.sum(scale * (expandedX + bias))` 累加。
3. 19 tiling key 分支对应 dtype×模板变体，Triton 中由 dtype constexpr + 循环策略处理。
4. 支持可选 x2 skip connection。
5. 首个 **moe** kernel_types 的 kernel 页。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。