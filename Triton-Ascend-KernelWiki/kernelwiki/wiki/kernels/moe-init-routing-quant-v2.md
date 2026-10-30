---
id: kernel-moe-init-routing-quant-v2
title: MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU
type: kernel
tags: [moe, init_routing, sort, gather, quant, dynamic_quant, drop, aiv_only, int8, int4, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [moe]
operators: [moe_init_routing_quant_v2]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-init-routing-quant-v2
related: [kernel-moe-init-routing-quant, kernel-moe-finalize-routing]
sources: [ops-transformer-moe-init-routing-quant-v2]
performance_claims: []
---

# MoeInitRoutingQuantV2（MoE 路由初始化 V2 + 动态量化 + drop）

CANN ops-transformer 的 `MoeInitRoutingQuantV2` 是 `MoeInitRoutingQuant` 的增强版（README L18-24）：新增 drop/dropless 模式、动态 quant、删除 rowIdx 输入、expertTokens 统计、INT4 输出、950(arch35) 支持。**AIV-only**（MIX_AIV_1_0）。

## 1. 输入输出

### 输入（def.cpp L23-46）
- x（REQUIRED，FP16/BF16/FP32）
- expert_idx（REQUIRED，INT32）
- scale / offset（OPTIONAL，FLOAT32）

### 输出（def.cpp L47-71）
- expanded_x（REQUIRED，INT8/INT4）
- expanded_row_idx（REQUIRED，INT32）
- expert_tokens_count_or_cumsum（OPTIONAL，INT32）
- expert_tokens_before_capacity（OPTIONAL，INT32）
- dynamic_quant_scale（OPTIONAL，FLOAT32）

### 属性（def.cpp L72-78）
`active_num`/`expert_capacity`/`expert_num`/`drop_pad_mode`(0=dropless,1=drop)/`expert_tokens_count_or_cumsum_flag`/`expert_tokens_before_capacity_flag`/`quant_mode`(0=静态,1=动态)。

## 2. 核心公式（README L25-83）

```
1. Sort: sortedExpertIdx, sortedRowIdx = keyValueSort(expertIdx)
2. SrcToDst: expandedRowIdx[sortedRowIdx[i]] = i
3. (dropless) expertTokensCumsum = Cumsum(Histogram(sortedExpertIdx))
   (drop)     expertTokensBeforeCapacity = Histogram(sortedExpertIdx)
4. Quant:
   静态: quantResult = round(x * scale + offset)
   动态: dynScale = row_max(abs(x[·scale])) / 127;  quantResult = round(x / dynScale)
5. Gather: expandedX[i] = quantResult[sortedRowIdx[i] % NUM_ROWS]
6. (drop) 容量截断/补零
```

## 3. tiling key dispatch（3 档）

| key | 模式 |
|---|---|
| 20000 | 静态 quant fullload（单阶段快速路径） |
| 21000 | 动态 quant fullload |
| 10000-11110 | 多阶段（Sort->ExpertToken->SrcToDst->GatherQuant） |

## 4. V2 vs V1 差异

| 差异 | V1 | V2 |
|---|---|---|
| rowIdx | 有 | 删除 |
| drop/dropless | 无 | 有 |
| 动态 quant | 无 | 有 |
| expertTokens 统计 | 无 | 有 |
| 架构 | A2/A3 | A2/A3 + 950 |
| 输出 dtype | INT8 | INT8/INT4 |

## 5. 架构支持（README L7-12）

Ascend910B1/910_93（arch32）+ Ascend950（arch35 regbase/simt）。不支持 310P/Kirin。

## 6. Agent actions

1. 识别为 MoE routing V2（drop/dropless + 静态/动态 quant + expertTokens 统计）。
2. 与 [[kernel-moe-init-routing-quant]] V1 区分：V2 删除 rowIdx、新增 drop 和动态 quant。
3. 动态 quant：`dynScale = row_max(abs(x[·scale])) / 127`，输出 dynamic_quant_scale。
4. drop 模式：按 expertCapacity 截断/补零。
5. arch35(950) 有 regbase/simt 变体（CANN 专有，不迁移）。
6. tiling key 20000/21000 是 fullload 快速路径（小数据量单阶段）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。