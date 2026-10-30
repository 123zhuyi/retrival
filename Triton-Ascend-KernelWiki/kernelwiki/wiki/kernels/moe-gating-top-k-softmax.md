---
id: kernel-moe-gating-top-k-softmax
title: MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU
type: kernel
tags: [moe, gating, softmax, topk, aiv_only, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [moe]
operators: [moe_gating_top_k_softmax]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-gating-top-k-softmax
related: [kernel-moe-init-routing-quant, kernel-moe-finalize-routing, kernel-moe-token-permute]
sources: [ops-transformer-moe-gating-top-k-softmax]
performance_claims: []
---

# MoeGatingTopKSoftmax（MoE gating softmax + topK）

CANN ops-transformer 的 `MoeGatingTopKSoftmax` 是 **MoE pipeline 的入口算子**：对 gating 输出 x 做 softmax 后取 topK，输出 topk 值(y)/专家序号(expertIdx)/行索引(rowIdx)（README L16）。是 [[kernel-moe-init-routing-quant]] 的上游（gating -> softmax -> topk -> init_routing -> FFN -> finalize）。**AIV-only**。

## 1. 输入输出

### 输入（def.cpp L23-34）
- x（REQUIRED，FP16/FP32/BF16）：gating 输出。
- finished（OPTIONAL，BOOL）：finished 标记。

### 输出（def.cpp L35-49）
- y（REQUIRED，FP16/FP32/BF16）：softmax 的 topk 值。
- expert_idx（REQUIRED，INT32）：topk 索引=专家序号。
- row_idx（REQUIRED，INT32）：行索引。

### 属性（def.cpp L50）
`k`（INT，topk 的 k 值）。

### 约束（README L93-96）
k≤1024；x 每维 < int32 max。

## 2. 核心公式（README L19-33）

```
1. softmaxOut = softmax(x, axis=-1)
2. yOut, expertIdxOut = topK(softmaxOut, k)
3. rowIdxOut = arange(N*K).reshape(K,N).transpose(1,0)
4. 若 finished[i]=True -> expertIdxOut = num_expert
```

## 3. tiling key dispatch（18 key，3 模板族）

| key | 模板族 | dtype | 说明 |
|---|---|---|---|
| 0-5 | E_K_FullLoad | FP32/FP16/BF16 × BUF(1/2) | 小数据量全装载 |
| 6-8 | K_FullLoad | FP32/FP16/BF16 × BUF(2) | 中数据量 k 全装载 |
| 9-17 | Perf | FP32/FP16/BF16 × ColRange(<8/8-64/>64) | 大数据量按列范围优化 |

## 4. 架构支持（README L7-12）

A2/A3(arch32) + Ascend950(arch35 regbase)。

## 5. MoE pipeline 位置

```
gating_top_k_softmax  ->  init_routing_quant  ->  FFN  ->  finalize_routing
   (本算子)              (排序+量化)            (专家)    (合并)
```

## 6. Agent actions

1. 识别为 MoE gating 入口（softmax + topk，AIV-only）。
2. 是 MoE pipeline 第一步，下游接 [[kernel-moe-init-routing-quant]]。
3. 3 模板族按数据量选择：E_K_FullLoad（小）/K_FullLoad（中）/Perf（大，按 expert 数 ColRange）。
4. Triton 中 topk 需自定义（无内置 topk）或 host 预排序。
5. finished 标记用于推理场景跳过已完成行。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。