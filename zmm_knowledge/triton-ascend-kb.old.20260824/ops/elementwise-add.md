---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: Elementwise add with a tail mask
description: Build one-dimensional offsets from the program id and block size. Use offsets < n on all loads and the final store. If accumulation or nonlinear work is sensitive, compute the intermediate value in FP32 before conversion.
tags: [elementwise, add, tail_mask, bf16, fp32_accumulation, ascend910b1, math]
confidence: inferred
status: active
resource: https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/elementwise-add.md
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/elementwise-add.md
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/elementwise-add.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/elementwise-add.md](https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/elementwise-add.md)。

# Elementwise add

Build one-dimensional offsets from the program id and block size. Use `offsets < n` on all loads and the final store. If accumulation or nonlinear work is sensitive, compute the intermediate value in FP32 before conversion.

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise 算子优化](../reference/op-class-elementwise.md) — 主题相关：共享 add, elementwise
- 实践案例: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../runbooks/algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 实践案例: [数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）](../runbooks/algorithm-numerics/math-normalize-precondition.md) — 同属优化知识，主题相关
- 实践案例: [整数比较/加法必须转 float32，避免 Vector 标量降级](../runbooks/compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [Triton-Ascend kernel 调试排查清单](../runbooks/meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../runbooks/algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 bf16
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 bf16
- 相关主题: [Element-wise 算子避免冗余 Reshape](../runbooks/compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 elementwise
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — elementwise 基础算子
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910b1, elementwise, fp32_accumulation
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910b1, elementwise, fp32_accumulation
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910b1, elementwise, fp32_accumulation
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation, tail_mask
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask

<!-- okf:related:end -->
