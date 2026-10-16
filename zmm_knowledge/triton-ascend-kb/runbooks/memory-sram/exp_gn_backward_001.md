---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: Tiled Two-Pass Reduction"
description: "normalization 场景下，Memory Access Optimization: Tiled Two-Pass Reduction，收益 35%"
tags: [访存效率, normalization-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: Tiled Two-Pass Reduction

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，group_size≥32、HW≥128；失效边界：HW 很大时两遍读取 grad_y 与 X 的 GM 流量翻倍，需结合软件流水掩盖延迟；HW<128 时 tile 数过少，部分和累加摊不薄两遍开销，单遍+原子规约可能更合适）

### 原则
- **Tiled Two-Pass Reduction**：The kernel separates the computation of reduction statistics (sum of gradients and weighted sum) from the final gradient application. This is necessary because the gradient formula requires global statistics over the spatial dimension. By tiling the spatial dimension with BLOCK_HW_SIZE, the kernel accumulates partial sums in registers/SRAM during the first pass, avoiding expensive atomic operations or global memory synchronization.
- **Spatial Tiling and Locality**：The loop over HW processes data in chunks of BLOCK_HW_SIZE (default 128). This optimizes for L2 cache and memory bandwidth by reusing loaded grad_y and X values within the accumulation registers (dx_part2, dx_part3) before moving to the next tile.
- **Precision Preservation**：Explicitly casts intermediate calculations to tl.float32 (e.g., rstd, mean, weight, dx_part2). This prevents numerical underflow or overflow during the reduction of variance and mean terms, which is critical for the stability of the backward pass in normalization layers.
- **Boundary-Aware Masking**：Utilizes dynamic masking (wb_mask, hw_mask, xy_mask) to handle arbitrary tensor dimensions where C or HW are not perfectly divisible by the block sizes. This ensures correctness for all input shapes without the overhead of padding.

### 代码/骨架出处
- `TritonPath`:
```python
group_norm_backward_kernel
```

**坏实践（反例）**：单遍内用原子加/GM 同步累加空间维全局统计量
```python
for hw0 in range(0, HW, BLOCK_HW_SIZE):
    part = tl.sum(dy * x, axis=...)          # tile 内部分和
    tl.atomic_add(stats_ptr + c, part)       # 每个 tile 原子写回全局统计
```
后果：大量原子操作与 GM 同步成为瓶颈；且统计量未全部就绪前无法应用梯度公式，正确性与性能双输。✅ 正解：两遍分离——第一遍按 BLOCK_HW_SIZE（默认 128）tile 在寄存器/UB 累加 partial sums 得到全局统计，第二遍再应用梯度公式；中间量（rstd、mean、weight、dx_part2）显式转 fp32 防 underflow/overflow；C/HW 不整除时用动态 mask 而非 padding。

**已知实例**：normalization §raw (`experiences/normalization/exp_gn_backward_001.yaml`)
- 声称性能收益：35%
- 备注：The kernel performs two passes over the input data (grad_y and X). For very large HW, this is memory bandwidth intensive. Optimizations could involve software pipelining within the loops to hide latency.

<!-- okf:related:start -->

# 相关

- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: npu simd vectorization](../compute-pipeline/exp_layernorm_npu_simd_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 normalization
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
