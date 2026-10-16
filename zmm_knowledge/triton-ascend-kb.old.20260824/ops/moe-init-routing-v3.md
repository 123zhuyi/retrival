---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU
description: CANN ops-transformer 中的 MoeInitRoutingV3 是 MoE routing 计算算子，V2 的升级版。支持不量化/动态量化（int8）/静态量化模式，输出 expandedx + expandedrowidx + experttokenscountorcumsum + expandedscale。纯 AIV-only Vector 算子（无 matmul），核心操作是 sort + 位置映射 + 直方图 + 量化 + gather。
tags: [moe, init_routing, sort, gather, quant, dynamic_quant, static_quant, aiv_only, fp32_accumulation, unified_buffer, register_file, vector_core, ascend910b1, ascend910_93, ascend950pr, moe_init_routing_v3]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_v3/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_v3/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-init-routing-v3.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_v3/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_init_routing_v3/)。

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

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-moe-init-routing-v3`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Gather/Scatter 算子优化](../reference/op-class-gather_scatter.md) — 主题相关：共享 gather, moe
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [Work-Stealing 动态负载均衡（不均匀任务分发）](../runbooks/compute-pipeline/work-stealing-dynamic-load-balance.md) — 主题相关：共享 moe
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](../runbooks/compute-pipeline/traj-gather_scatter-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Gather/Scatter 内存访问优化聚合](../runbooks/memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 moe
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, dynamic_quant, fp32_accumulation, gather, init_routing, moe, quant, sort, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, init_routing, moe, quant, sort, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, init_routing, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — MoE init routing V2/V3 版本迭代
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, gather, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, sort, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core

<!-- okf:related:end -->
