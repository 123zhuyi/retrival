---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU
description: CANN ops-transformer 中的 MoeFinalizeRoutingV2Grad 算子是 MoeFinalizeRoutingV2 的反向传播算子。核心是 expandedRowIdx 驱动的 scatter-add（gradY 按索引映射到 gradExpandedXOut）+ 可选 scale 梯度计算。三架构（A2/A3/950），纯 Vector 计算，无 Cube。
tags: [moe_finalize_routing_v2_grad, moe, routing, backward, scatter_add, elementwise, reduce, fp32_accumulation, unified_buffer, register_file, vector_core, ascend910b1, ascend910_93, ascend950pr]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing_v2_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing_v2_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-finalize-routing-v2-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing_v2_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_finalize_routing_v2_grad/)。

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

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、atomic_add 多核精度边界需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-moe-finalize-routing-v2-grad`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Embedding / 索引拷贝算子优化](../reference/op-class-embedding.md) — 主题相关：共享 backward
- 实践案例: [Gather/Scatter 算子优化](../reference/op-class-gather_scatter.md) — 主题相关：共享 moe, routing
- 实践案例: [RoPE（旋转位置编码）算子优化](../reference/op-class-rope.md) — 主题相关：共享 backward
- 实践案例: [跨 tile 归约必须显式两阶段 reduce-then-apply](../runbooks/algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [Add算子块大小动态调整优化经验](../runbooks/memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [带 alpha 缩放的逐元素 add 访存与块大小优化](../runbooks/memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 elementwise
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 elementwise
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 elementwise
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950pr, elementwise, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, elementwise, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 elementwise
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 同一 MoeFinalizeRoutingV2 算子的前向与反向实现
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, routing, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, routing, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, moe, reduce, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, reduce, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, elementwise, fp32_accumulation, reduce, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
