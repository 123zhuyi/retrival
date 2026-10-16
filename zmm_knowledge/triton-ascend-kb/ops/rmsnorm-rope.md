---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: RMSNorm-RoPE fused kernel on Ascend 910B1
description: Accumulate the sum of squares in FP32 and convert only at the output boundary. Keep the RoPE even/odd channel pairing explicit. For the documented small-batch, hidden-4096 case, inspect whether lanes access adjacent addresses before chan...
tags: [rmsnorm, rope, normalization, vecutil_low, bf16, load_order, contiguous_access, fp32_accumulation, unified_buffer, vector_core, ascend910b1, norm, posembedding]
confidence: inferred
status: active
resource: https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rmsnorm-rope.md
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rmsnorm-rope.md
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/rmsnorm-rope.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rmsnorm-rope.md](https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rmsnorm-rope.md)。

# RMSNorm + RoPE

Accumulate the sum of squares in FP32 and convert only at the output boundary. Keep the RoPE even/odd channel pairing explicit. For the documented small-batch, hidden-4096 case, inspect whether lanes access adjacent addresses before changing tile size.

## Agent actions

1. Preserve RMSNorm and RoPE semantics.
2. Check UB use for input, weight, FP32 temporaries, and output.
3. Apply a tail mask when the hidden dimension is not tile-aligned.
4. Compare the current load order with the contiguous hidden dimension.
5. Validate correctness before accepting a profiling improvement.

<!-- okf:related:start -->

# 相关

- 实践案例: [融合算子综合页](../reference/op-class-fused_ops.md) — 主题相关：共享 rmsnorm, rope
- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 normalization, rmsnorm
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rmsnorm, rope
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../runbooks/memory-sram/exp_fused_rms_norm_001.md) — 主题相关：共享 normalization
- 实践案例: [Memory Access Optimization: fused split and dual-norm single-pass load](../runbooks/memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization
- 实践案例: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../runbooks/memory-sram/exp_gemma_rmsnorm_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](../runbooks/memory-sram/exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization
- 实践案例: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../runbooks/memory-sram/parallel-axis-align.md) — 主题相关：共享 contiguous_access
- 相关主题: [Rmsnorm Fused Skeleton](../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 bf16
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 bf16, rope
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 bf16
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, bf16, contiguous_access, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910b1, contiguous_access, fp32_accumulation, rmsnorm, rope, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910b1, fp32_accumulation, rmsnorm, rope, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910b1, fp32_accumulation, rmsnorm, rope, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910b1, bf16, contiguous_access, fp32_accumulation, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — RoPE 相关算子/指针处理模式

<!-- okf:related:end -->
