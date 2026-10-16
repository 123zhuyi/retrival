---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: adaptive tiling strategy"
description: "normalization 场景下，Memory Access Optimization: adaptive tiling strategy，收益 25%"
tags: [访存效率, normalization-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: adaptive tiling strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，n_cols ≥4096、n_rows ≥32 的大列归一化 backward；失效边界：n_cols 较小、单 tile 可放下时两遍列循环的重载开销大于收益；n_rows 极小时 dW/dB 的 atomic_add 争用成为瓶颈）

### 原则
- **adaptive tiling strategy**：The kernel utilizes a heuristic to dynamically determine BLOCK_SIZE_M based on n_cols (specifically `ceil_div(4096, n_cols)`). This ensures that the total tile size (BLOCK_SIZE_M * BLOCK_SIZE_N) remains relatively constant, optimizing SRAM/L1 cache utilization regardless of the matrix aspect ratio.
- **fused accumulation**：Gradients for weights (dW) and bias (dB) are accumulated in registers (dW_acc, dB_acc) inside the main loop over rows. This fuses the reduction operation with the element-wise computation, significantly reducing global memory traffic by avoiding intermediate writes.
- **grid stride loop**：The workload distribution uses a grid-stride loop (`for row_task_id in range(pid, num_row_tasks, num_programs)`). This allows the kernel to efficiently handle any number of rows by load-balancing across all available program instances without requiring the grid size to exactly match the problem size.
- **atomic reduction resolution**：The final accumulation of dW and dB is handled via `tl.atomic_add`. This allows multiple program instances to safely update the same gradient memory locations in parallel, eliminating the need for a separate reduction kernel or expensive global synchronization.
- **two pass reduction strategy**：The kernel implements a two-pass algorithm over the column dimension. The first pass iterates over tiles to compute the reduction statistics (c1 and c2) required for the DX gradient. The second pass iterates again to compute the final DX, DW, and DB using these statistics. This avoids materializing the full intermediate x_hat matrix, significantly reducing memory bandwidth requirements for large column sizes.
- **column major tiling**：To handle the "large cols" scenario where n_cols exceeds the SRAM capacity, the kernel processes the column dimension in chunks of BLOCK_SIZE_N. This tiling strategy ensures that data loaded into the cache is reused for both the reduction phase and the gradient computation phase, optimizing data locality.
- **parallel atomic accumulation**：The kernel utilizes tl.atomic_add for accumulating DW and DB gradients. This allows different program 实例 (processing different rows) to write to the same global memory locations for weights and biases without synchronization, enabling high parallelism across the row dimension.
- **mixed precision accumulation**：Intermediate accumulators (c1_acc, c2_acc) and computations are explicitly cast to tl.float32. This preserves numerical precision during the reduction operations, which is critical for the stability of the backward pass, before casting the final DX result back to the original input dtype.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/237-_layernorm_bwd_large_cols_kernel.py
```

**坏实践（反例）**：物化整个中间归一化矩阵，dW/dB 用单独的 reduction kernel 收尾
```python
x_hat = normalize(x)                  # 中间结果整块写回 GM
tl.store(x_hat_ptr, x_hat)
dw_db_kernel(...)                     # 单独 kernel 串行归约 dW/dB
dx_kernel(...)                        # 再从 GM 读回 x_hat 算 dX
```
后果：大列场景下中间矩阵的 GM 往返读写成为带宽瓶颈；额外 kernel launch 与全局同步开销叠加。✅ 正解：两遍列循环（第一遍算统计量 c1/c2，第二遍算 dX/dW/dB），不物化中间矩阵；dW/dB 在循环内寄存器累加，收尾用 `tl.atomic_add` 并行归约；BLOCK_SIZE_M 用 `ceil_div(4096, n_cols)` 自适应。

**已知实例**：normalization §raw (`experiences/normalization/exp_layernorm_bwd_001.yaml`)
- 声称性能收益：25%
- 备注：专为列数（hidden size）很大的归一化 backward 优化。行数极小时 dW/dB 上的原子操作可能成为瓶颈。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: npu simd vectorization](../compute-pipeline/exp_layernorm_npu_simd_001.md) — 主题相关：共享 normalization, normalization-operations
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
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](ub-aware-block-adaptive.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
