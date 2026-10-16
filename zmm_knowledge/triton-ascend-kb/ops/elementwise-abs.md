---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: Elementwise absolute value (Abs)
description: Elementwise absolute value (Abs)
tags: [elementwise, abs, tail_mask, contiguous_access, aiv_only, bf16, complex, dtype_dispatch, fp32_accumulation, ascend910b1, ascend950pr, ascend950dt, math]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-math/tree/b56b2db0ebc3fcc96f5f4c7f3060b4375481175e/math/abs/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-math/tree/b56b2db0ebc3fcc96f5f4c7f3060b4375481175e/math/abs/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/elementwise-abs.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-math/tree/b56b2db0ebc3fcc96f5f4c7f3060b4375481175e/math/abs/](https://gitcode.com/cann/ops-math/tree/b56b2db0ebc3fcc96f5f4c7f3060b4375481175e/math/abs/)。

# Elementwise absolute value (Abs)

## Semantic

`y = abs(x)` — element-wise absolute value. Output shape equals input shape.

## Data Type Handling

### Real dtype

| Input dtype | Output dtype | Compute dtype | Intermediate |
|-------------|--------------|---------------|---------------|
| float32     | float32      | float32       | none |
| float16     | float16      | float16       | none |
| bfloat16    | bfloat16     | float32       | FP32 intermediate |
| int8/16/32/64 | same       | same          | none |
| uint8/bool  | same         | passthrough   | identity (no-op) |

### Complex dtype

| Input dtype | Output dtype | Formula |
|-------------|--------------|---------|
| COMPLEX32   | float16      | `sqrt(real^2 + imag^2)` |
| COMPLEX64   | float32      | `sqrt(real^2 + imag^2)` with numerical stability |

For COMPLEX64, the CANN kernel uses a numerically stable algorithm:
1. Sort `|real|` and `|imag|` into max/min
2. Normalize to [1, 2) range to avoid overflow
3. Compute `sqrt(norm_max^2 + norm_min^2)` and multiply back exponent
4. Handle subnormals and zero separately

## Tiling Key Dispatch

| Key | Condition | Description |
|-----|-----------|-------------|
| 101 | Output dtype == BF16 | BF16 input, FP32 intermediate, BF16 output |
| 102 | Real dtype, not BF16 | Direct same-dtype abs |
| 103 | COMPLEX32 or COMPLEX64 | Complex abs with promoted output |

## Kernel Pipeline

### Real dtype DAG

```
CopyIn → Cast → Abs → Cast → CopyOut
```

For BF16 (key 101): `CopyIn<bfloat16> → Cast<float, bfloat16, 0> → Abs<float> → Cast<bfloat16, float, 1> → CopyOut<bfloat16>`

The FP32 intermediate cast prevents precision loss in the abs operation.

For other real dtypes (key 102): Cast nodes are identity (same dtype in/out); the pipeline reduces to `CopyIn → Abs → CopyOut`.

### Complex dtype DAG

```
CopyIn → AbscomplexCustom → CopyOut
```

The custom AbscomplexCustom operation handles interleaved real/imag pairs with the numerical stability algorithm described above.

## AIV-only Execution

The kernel is marked `KERNEL_TYPE_AIV_ONLY` — it runs on vector (AIV) cores only, not on cube (AIC) cores. This is the standard dispatch for elementwise vector math.

## Triton-Ascend Skeleton

The key Triton primitives used for an elementwise abs kernel:

1. `tl.program_id(0)` — 1D grid index
2. `BLOCK_SIZE` — tile width
3. Offsets: `offsets = start + tl.arange(0, BLOCK_SIZE)`
4. Tail mask: `mask = offsets < n_elements`
5. `tl.load(x_ptr + offsets, mask=mask, other=0.0)`
6. `tl.abs(x)` or `tl.where(x < 0, -x, x)` for compute
7. `tl.store(y_ptr + offsets, result, mask=mask)`

For BF16, load as BF16, cast to FP32, compute abs, cast back to BF16, store.

For complex, assume interleaved `[real0, imag0, real1, imag1, ...]` layout — load pairs, compute `tl.sqrt(real*real + imag*imag)` manually.

## References

- Source: `source-cann-ops-math-abs` — full CANN ops-math Abs operator analysis
- Tail mask: `technique-tail-mask` — boundary predicate pattern
- Contiguous load: `technique-contiguous-load-order` — neighboring lane mapping
- FP32 accumulation: `technique-fp32-accumulation` — BF16 intermediate precision
- Elementwise pattern: `kernel-elementwise-add` — sibling elementwise kernel

<!-- okf:related:start -->

# 相关

- 实践案例: [数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）](../runbooks/algorithm-numerics/math-normalize-precondition.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [Memory Access Optimization: fp8 value casting](../runbooks/memory-sram/exp_attn_fp8_ub_slice_001.md) — 同属优化知识，主题相关
- 实践案例: [Add算子块大小动态调整优化经验](../runbooks/memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [带 alpha 缩放的逐元素 add 访存与块大小优化](../runbooks/memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../runbooks/memory-sram/parallel-axis-align.md) — 主题相关：共享 contiguous_access
- 实践案例: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](../runbooks/portability/exp_rope_llama4_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 aiv_only, bf16, elementwise
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 aiv_only, bf16, elementwise
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 bf16
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — elementwise 基础算子
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 aiv_only, elementwise
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910b1, contiguous_access, fp32_accumulation, tail_mask
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, fp32_accumulation
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910b1, ascend950pr, elementwise, fp32_accumulation
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, elementwise, fp32_accumulation
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, bf16, fp32_accumulation
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, bf16, fp32_accumulation
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, fp32_accumulation
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, contiguous_access, elementwise, fp32_accumulation, tail_mask
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, bf16, elementwise, fp32_accumulation
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, contiguous_access, fp32_accumulation
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask

<!-- okf:related:end -->
