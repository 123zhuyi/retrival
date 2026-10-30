---
id: kernel-moe-init-routing-quant
title: MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU
type: kernel
tags: [moe, init_routing, sort, gather, quant, aiv_only, int8, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [moe]
operators: [moe_init_routing_quant]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-init-routing-quant
related: [kernel-moe-finalize-routing]
sources: [ops-transformer-moe-init-routing-quant]
performance_claims: []
---

# MoeInitRoutingQuant（MoE 路由初始化 + 量化）

CANN ops-transformer 的 `MoeInitRoutingQuant` 是 **MoE 路由初始化 + 量化**算子：根据 gating topk 结果做 routing 处理并量化为 INT8（README L16）。3 阶段：Sort -> SrcToDst -> GatherOut+Quant。**AIV-only**（无 Cube）。与 [[kernel-moe-finalize-routing]] 配对（init 量化输入，finalize 合并输出）。

## 1. 输入输出

### 输入（def.cpp L23-40）
- x（REQUIRED，FP16/BF16/FP32）：token 特征。
- row_idx（REQUIRED，INT32）：原始行位置。
- expert_idx（REQUIRED，INT32）：每行对应的 K 个 expert。

### 输出（def.cpp L41-55）
- expanded_x（REQUIRED，INT8）：扩展并量化的特征。
- expanded_row_idx（REQUIRED，INT32）：expandedX 与 x 的映射。
- expanded_expert_idx（REQUIRED，INT32）：排序后的 expertIdx。

### 属性（def.cpp L56-58）
`active_num`（INT）、`scale`（FLOAT）、`offset`（FLOAT）。

## 2. 核心公式（README L19-29）

```
1. Sort: expandedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)
2. SrcToDst: expandedRowIdx[sortedRowIdx[i]] = i
3. Gather+Quant: expandedX[i] = quant(x[sortedRowIdx[i] % numRows], scale, offset)
```

## 3. tiling key dispatch（4 key）

| key | 排序 | gather |
|---|---|---|
| 1 | OneCore | Normal |
| 2 | MultiCore | Normal |
| 3 | OneCore | SmallActiveRow |
| 4 | MultiCore | SmallActiveRow |

## 4. 架构支持（README L7-12；def.cpp L59-60）

仅 Ascend910B1/910_93（A2/A3）。不支持 950/310P/Kirin/A3。

## 5. Agent actions

1. 识别为 MoE 路由初始化+量化（AIV-only，3 阶段 sort+scatter+gather+quant）。
2. 与 [[kernel-moe-finalize-routing]] 配对：本算子量化输入供 FFN，finalize 合并 FFN 输出。
3. 量化：`expandedX = round(x * scale + offset)` cast to INT8。
4. Triton 中 sort 需自定义（如 bitonic）或 host 预排序。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。