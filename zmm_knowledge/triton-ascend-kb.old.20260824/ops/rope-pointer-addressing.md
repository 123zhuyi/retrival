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

- 实践案例: [融合算子综合页](../reference/op-class-fused_ops.md) — 主题相关：共享 rope
- 实践案例: [RoPE（旋转位置编码）算子优化](../reference/op-class-rope.md) — 主题相关：共享 rope
- 实践案例: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rope
- 实践案例: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../runbooks/compute-pipeline/scalar-load-channel-index.md) — 同属优化知识，主题相关
- 实践案例: [Triton-Ascend kernel 调试排查清单](../runbooks/meta-skills/debugging-checklist.md) — 主题相关：共享 compile_error
- 实践案例: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../runbooks/portability/cdiv-kernel-vs-host.md) — 主题相关：共享 compile_error
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../runbooks/compute-pipeline/traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](../runbooks/compute-pipeline/traj-rope-compute.md) — 主题相关：共享 rope
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../runbooks/compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 rope
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../runbooks/memory-sram/traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../runbooks/memory-sram/traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../runbooks/memory-sram/traj-rope-memory_access.md) — 主题相关：共享 rope
- 相关主题: [Triton-Ascend 调试排查典型问题聚合](../runbooks/meta-skills/traj-general_debug-debugging.md) — 主题相关：共享 compile_error
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](../runbooks/portability/traj-rope-hardware_adaptation.md) — 主题相关：共享 rope
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910b1, rope, tail_mask
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910b1, rope
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910b1, rope, tail_mask
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — RoPE 相关算子/指针处理模式

<!-- okf:related:end -->
