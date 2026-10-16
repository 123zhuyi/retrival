---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU
description: CANN ops-transformer 的 MoeInitRoutingQuantV2 是 MoeInitRoutingQuant 的增强版（README L18-24）：新增 drop/dropless 模式、动态 quant、删除 rowIdx 输入、expertTokens 统计、INT4 输出、950(arch35) 支持。AIV-only（MIXAIV10）。
tags: [moe, init_routing, sort, gather, quant, dynamic_quant, drop, aiv_only, int8, int4, fp16, bf16, fp32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, ascend950pr, moe_init_routing_quant_v2]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_quant_v2/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_quant_v2/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-init-routing-quant-v2.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_quant_v2/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_quant_v2/)。

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
2. 与 `kernel-moe-init-routing-quant` V1 区分：V2 删除 rowIdx、新增 drop 和动态 quant。
3. 动态 quant：`dynScale = row_max(abs(x[·scale])) / 127`，输出 dynamic_quant_scale。
4. drop 模式：按 expertCapacity 截断/补零。
5. arch35(950) 有 regbase/simt 变体（CANN 专有，不迁移）。
6. tiling key 20000/21000 是 fullload 快速路径（小数据量单阶段）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Gather/Scatter 算子优化](../reference/op-class-gather_scatter.md) — 主题相关：共享 gather, moe
- 实践案例: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../runbooks/algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 实践案例: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../runbooks/algorithm-numerics/fp32-reduction-accumulator.md) — 同属优化知识，主题相关
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 实践案例: [Work-Stealing 动态负载均衡（不均匀任务分发）](../runbooks/compute-pipeline/work-stealing-dynamic-load-balance.md) — 主题相关：共享 moe
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../runbooks/meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 主题相关：共享 int8
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../runbooks/algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Activation Scheduling 优化聚合](../runbooks/compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 fp32
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](../runbooks/compute-pipeline/traj-gather_scatter-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Gather/Scatter 内存访问优化聚合](../runbooks/memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 moe
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, bf16, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — MoE init routing 量化版本迭代
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, init_routing, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, dynamic_quant, fp32_accumulation, gather, init_routing, moe, quant, sort, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, int8, moe, sort, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, int8, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, bf16, fp16, fp32, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
