---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: two pass reduction strategy"
description: "softmax 场景下，Memory Access Optimization: two pass reduction strategy，收益 30%"
tags: [访存效率, activation-operations, softmax]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: two pass reduction strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，N ≥128；失效边界：N 超单 tile 容量时才需要两遍结构，小 N 应走 ONE_TILE_PER_CTA 快路径；evict_last 只适用于两遍间会复用的张量，对不再复用的数据误用会污染缓存）

### 原则
- **two pass reduction strategy**：The kernel decouples the computation into two distinct loops a reduction phase to compute the scale (sum of output * gradient) and an update phase to calculate the input gradient. This approach avoids the need to store intermediate results for the entire row in registers or scratchpad, which is critical for handling large N dimensions that exceed register capacity.
- **explicit cache eviction policy**：In the reduction loop, the `eviction_policy="evict_last"` argument is used for the `out` tensor. This explicitly instructs the memory subsystem to retain the loaded `out` data in the cache (L1/L2), as it will be accessed again in the subsequent update loop. This optimization significantly reduces global memory bandwidth pressure by maximizing data reuse across the two passes.
- **adaptive tiling and heuristics**：The kernel utilizes `triton.autotune` and runtime heuristics to dynamically select the optimal `TILE_M` and `TILE_N` based on the input shapes M and N. This ensures that the memory access patterns 沿连续维对齐 GM→UB 向量访存 and that NPU 核占用率 is maximized for varying batch sizes and sequence lengths.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/129-softmax_backward_kernel_inner.py
```

**坏实践（反例）**：单遍法把整行的中间结果（输出与梯度的逐元素积）驻留寄存器，同时算 scale 和 dx
```python
row = tl.load(out_ptr + offs)         # 整行驻留寄存器
g   = tl.load(grad_ptr + offs)
scale = tl.sum(row * g)               # 归约依赖未完成前无法释放 row
```
后果：N 超过寄存器容量时编译器 spill 或 kernel 不可用。✅ 正解：两遍解耦——第一遍只归约出 scale（fp32 标量），第二遍重读数据算 dx；第一遍对 `out` 用 `eviction_policy="evict_last"` 保留在缓存中供第二遍复用。

**已知实例**：softmax §raw (`experiences/softmax/exp_softmax_backward_001.yaml`)
- 声称性能收益：30%
- 备注：The `ONE_TILE_PER_CTA` path provides a fast-path for smaller N dimensions where the data fits in a single tile, bypassing the overhead of the two-pass loop structure.

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 softmax
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 softmax
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 softmax
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 softmax
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 softmax
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gating_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 主题相关：共享 softmax
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](../compute-pipeline/exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, softmax, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations, 访存效率

<!-- okf:related:end -->
