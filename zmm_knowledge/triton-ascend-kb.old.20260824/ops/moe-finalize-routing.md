---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU
description: CANN ops-transformer 的 MoeFinalizeRouting 是 MoE 最终路由合并算子：将 MoE FFN 的输出按 routing 索引和 scale 加权合并回原始 token 位置，支持 skip connection（x1+x2）和 expert bias（README L18-19）。纯 AIV-only elementwise 算子（gather+scale+add，无 Cube/matmul 参与）。
tags: [moe, finalize_routing, elementwise, gather, scale_add, aiv_only, fp16, bf16, fp32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, kirin9030, kirinx90, moe_finalize_routing]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-finalize-routing.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing/)。

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

<!-- okf:related:start -->

# 相关

- 实践案例: [Gather/Scatter 算子优化](../reference/op-class-gather_scatter.md) — 主题相关：共享 gather, moe
- 实践案例: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../runbooks/algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 实践案例: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../runbooks/algorithm-numerics/fp32-reduction-accumulator.md) — 同属优化知识，主题相关
- 实践案例: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](../runbooks/compute-pipeline/grid-by-output-count.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../runbooks/meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../runbooks/algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Activation Scheduling 优化聚合](../runbooks/compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 fp32
- 相关主题: [Element-wise 算子避免冗余 Reshape](../runbooks/compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 elementwise
- 相关主题: [Gather/Scatter 内存访问优化聚合](../runbooks/memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 moe
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — MoE finalize routing 的版本迭代
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, gather, kirin9030, kirinx90, moe, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, gather, moe, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, elementwise, fp16, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, elementwise, fp16, fp32, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
