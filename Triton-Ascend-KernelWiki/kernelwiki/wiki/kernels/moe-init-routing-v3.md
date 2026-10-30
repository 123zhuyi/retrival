---
id: kernel-moe-init-routing-v3
title: MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU
type: kernel
tags: [moe, init_routing, sort, gather, quant, dynamic_quant, static_quant, aiv_only]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [moe]
operators: [moe_init_routing_v3]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-init-routing-v3
related: []
sources: [ops-transformer-moe-init-routing-v3]
performance_claims: []
---

# MoeInitRoutingV3 (MoE routing 含动态/静态量化)

CANN ops-transformer 中的 `MoeInitRoutingV3` 是 MoE routing 计算算子，V2 的升级版。支持不量化/动态量化（int8）/静态量化模式，输出 expanded_x + expanded_row_idx + expert_tokens_count_or_cumsum + expanded_scale。纯 AIV-only Vector 算子（无 matmul），核心操作是 sort + 位置映射 + 直方图 + 量化 + gather。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| x | 必选 | token 特征输入 | INT8/FP16/BF16/FLOAT（950 +HIFLOAT8）|
| expert_idx | 必选 | 每行对应的 K 个专家 id | INT32 |
| scale | 可选 | 量化参数 | FLOAT |
| offset | 可选 | 量化偏移 | FLOAT |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| expanded_x | 排序+量化后的展开 x | INT8/FP16/BF16/FLOAT（950 +FP8/HIFLOAT8）|
| expanded_row_idx | 行索引映射 | INT32 |
| expert_tokens_count_or_cumsum | 专家 token 计数/累计和 | INT64 |
| expanded_scale | 量化 scale 输出 | FLOAT（950 +FP8_E8M0）|

### 属性
`active_num`、`expert_capacity`、`expert_num`、`drop_pad_mode`、`expert_tokens_num_type`、`expert_tokens_num_flag`、`quant_mode`、`active_expert_range`、`row_idx_type`

## 2. 核心公式（6 步）
1. `sortedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)`
2. `expandedRowIdxOut[sortedRowIdx[i]] = i`
3. drop 模式：`expertTokensCountOrCumsum = Histogram(sortedExpertIdx)`
4. 量化：动态 `row_max(abs(x))/127` + round；静态 `scale/offset`
5. `expandedXOut[i] = quantResult[sortedRowIdx[i] % NUM_ROWS]`
6. `availableIdxNum = |{x ∈ expertIdx | expert_start ≤ x < expert_end}|`

## 3. 模板 dispatch

`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。~30 tiling key：
- **FullLoad 高性能**（2000000 系列）：UNQUANTIZED/STATIC_QUANT/DYNAMIC_QUANT × GATHER/SCATTER
- **非 FullLoad**：SortOneCore/SortMultiCore × Gather/Scatter × NonQuant/DynamicQuant/StaticQuant × NoDrop/Drop

## 4. V2 -> V3 变更（README.md）
1. 增加动态量化，支持 int8 动态量化输出
2. 增加 activeExpertRange，筛选有效 expertId
3. 删除 expertTokensBeforeCapacityFlag/输出，改用 expertTokensCountOrCumsumOut

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-init-routing-v3]]。