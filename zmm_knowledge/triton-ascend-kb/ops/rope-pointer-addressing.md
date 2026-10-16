---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: RoPE paired-channel pointer addressing
description: Derive pairbase from a non-negative channel index, then add the lane within the pair. Do not form a pointer with currentoffset - 1; use a boundary predicate for incomplete pairs.
tags: [rope, pointer_rewrite, compile_error, tail_mask, ascend910b1, posembedding]
confidence: inferred
status: active
resource: https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rope-pointer-addressing.md
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rope-pointer-addressing.md
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/rope-pointer-addressing.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rope-pointer-addressing.md](https://gitcode.com/gcw_85ZVbQ4M/Triton-Ascend-KernelWiki/blob/7d6a67e7f71d1bfb7cc51f66ae139c34ace40e9d/kernelwiki/wiki/kernels/rope-pointer-addressing.md)。

# RoPE pointer addressing

Derive `pair_base` from a non-negative channel index, then add the lane within the pair. Do not form a pointer with `current_offset - 1`; use a boundary predicate for incomplete pairs.

<!-- okf:related:start -->

# 相关

- 实践案例: [Compute Optimization: pointer base precompute](../runbooks/compute-pipeline/exp_causal_conv1d_prefill_004.md) — 同属优化知识，主题相关
- 实践案例: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../runbooks/compute-pipeline/scalar-load-channel-index.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: block pointer abstraction](../runbooks/memory-sram/exp_attn_ptr_setup_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: block pointer utilization](../runbooks/memory-sram/exp_idx_copy_001.md) — 同属优化知识，主题相关
- 实践案例: [Triton-Ascend kernel 调试排查清单](../runbooks/meta-skills/debugging-checklist.md) — compile_error 修复实例（指针寻址边界谓词）
- 实践案例: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../runbooks/portability/cdiv-kernel-vs-host.md) — 主题相关：共享 compile_error
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910b1
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910b1
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910b1
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910b1, rope, tail_mask
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910b1, rope
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910b1, rope, tail_mask
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — RoPE 相关算子/指针处理模式
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910b1, tail_mask

<!-- okf:related:end -->
