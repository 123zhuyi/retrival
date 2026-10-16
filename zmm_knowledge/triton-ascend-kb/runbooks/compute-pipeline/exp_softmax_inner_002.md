---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized accumulation strategy'
description: 'softmax 场景下，Compute Optimization: vectorized accumulation strategy，收益 30%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, softmax]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized accumulation strategy

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：N 维很大、循环内水平归约开销不可接受时收益最大（备注自述）；N < TILE_N 时单块即可，向量化累计的多余宽度浪费；evict_first 只适用于写完不再读的数据）

### 原则
- **vectorized accumulation strategy**：The kernel employs a vectorized reduction strategy where intermediate max (m) and sum (z) values are maintained as vectors of size TILE_N instead of scalars within the reduction loop. This avoids expensive horizontal reduction operations (e.g., tl.max) inside the hot loop, replacing them with efficient element-wise operations (tl.maximum). The final reduction to a scalar is performed only once after the loop completes.
- **online softmax algorithm**：Implements the numerically stable online softmax algorithm adapted for the vectorized strategy. It updates the running max and sum using the formula z = z * exp(m_old - m_new) + exp(x - m_new), ensuring stability against overflow/underflow for large input values.
- **cache eviction policy**：Utilizes eviction_policy="evict_first" in the second pass (write-back). This optimizes L1 cache behavior by treating input data loads as transient, preventing them from polluting the cache and evicting the accumulator values or other critical data during the final normalization phase.
- **loop specialization and tiling**：The kernel tiles the reduction dimension N into blocks of TILE_N. It specializes the loop logic for the main body and the boundary (remainder) tiles. This removes redundant mask checks from the main loop body, significantly improving instruction throughput and reducing branch mispredictions.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/127-softmax_kernel_inner.py
```

**坏实践（反例）**：归约循环内每次迭代都做水平归约（`tl.max`/`tl.sum` 出标量）→ 热循环被横向归约指令主导 → ✅ 正解即本条：max/sum 以 TILE_N 向量形式在循环内逐元素更新（`tl.maximum`），循环结束后才归约一次出标量。

**已知实例**：softmax §raw (`experiences/softmax/exp_softmax_inner_002.yaml`)
- 声称性能收益：30%
- 备注：This optimization is particularly effective for large N dimensions where the cost of horizontal reductions in the loop would be prohibitive. The vectorized approach leverages the full width of the vector units.

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 softmax
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 softmax
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 softmax
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 softmax
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 softmax
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_tiled_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减

<!-- okf:related:end -->
