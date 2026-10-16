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
- 实践案例: [Compute Optimization: dim is multiple mask elision](../runbooks/compute-pipeline/exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 实践案例: [Compute Optimization: dim is multiple mask elimination](../runbooks/compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 实践案例: [Compute Optimization: redundant mask elimination by matrix property](../runbooks/compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 实践案例: [整数比较/加法必须转 float32，避免 Vector 标量降级](../runbooks/compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [Add算子块大小动态调整优化经验](../runbooks/memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [带 alpha 缩放的逐元素 add 访存与块大小优化](../runbooks/memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../runbooks/memory-sram/exp_fused_rms_norm_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../runbooks/memory-sram/exp_gemma_rmsnorm_001.md) — 同属优化知识，主题相关
- 实践案例: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../runbooks/memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: dynamic block sizing with fused backward+add](../runbooks/memory-sram/exp_rmsnorm_bwd_fused_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](../runbooks/memory-sram/exp_rmsnorm_fused_quant_001.md) — 同属优化知识，主题相关
- 实践案例: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../runbooks/memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 实践案例: [Triton-Ascend kernel 调试排查清单](../runbooks/meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 实践案例: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../runbooks/meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 实践案例: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../runbooks/meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 实践案例: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../runbooks/portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 实践案例: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](../runbooks/portability/exp_rope_llama4_npu_001.md) — 同属优化知识，主题相关
- 实践案例: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../runbooks/portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 实践案例: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../runbooks/portability/triton-ascend-mask-codegen-traps.md) — 同属优化知识，主题相关
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 add, bf16, elementwise
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 bf16, elementwise
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 bf16
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — elementwise 基础算子
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 elementwise
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
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910b1, elementwise, fp32_accumulation
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation, tail_mask
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask

<!-- okf:related:end -->
