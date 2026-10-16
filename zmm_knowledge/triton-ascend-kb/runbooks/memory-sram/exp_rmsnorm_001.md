---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: grid stride scheduling"
description: "normalization 场景下，Memory Access Optimization: grid stride scheduling，收益 15%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: grid stride scheduling

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，n_rows ≥1、n_cols ≥1024 的 RMS 归一化类 kernel；失效边界：n_cols 较小、单行可整体驻留寄存器/UB 时，两遍法多读一遍 X 成为净损失，应改单遍法；n_rows < NUM_CORES 时 grid-stride 退化为单轮，调度收益稀释）

### 原则
- **grid stride scheduling**：Implements a grid-stride loop (for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES)) to map a fixed number of programs to a potentially large number of rows. This ensures high utilization of all available compute cores (NUM_CORES) without launching excessive grid dimensions, reducing kernel launch overhead and improving load balancing.
- **vectorized block access**：Utilizes 2D block loading (BLOCK x N_COLS) via advanced indexing (pos_offset[:, None] * stride + dim_range[None, :]). This maximizes global memory bandwidth by ensuring contiguous GM→UB access patterns and reducing the number of memory transactions.
- **mixed precision accumulation**：Casts input data to float32 (xf = x.to(tl.float32)) before computing the sum of squares. This prevents overflow/underflow during the variance calculation, which is critical for stability in fp16/bf16 inputs, before casting back to the original dtype.
- **weight broadcasting**：Loads the weight tensor W once per kernel instance and broadcasts it to the shape (BLOCK, N_COLS). This avoids repeated global memory reads of the weight vector for every row in the block, significantly reducing memory traffic for the scaling operation.
- **persistent kernel scheduling**：The kernel implements a grid-stride loop (`for row_idx in range(pid, n_rows, NUM_CORES)`), allowing a single program ID to process multiple rows sequentially. This reduces kernel launch overhead and improves hardware utilization by keeping the compute units busy.
- **weight vector reuse**：The weight vector `w` is loaded once outside the row processing loop. This optimization significantly reduces global memory bandwidth pressure by caching the weights in registers or on-chip memory for reuse across all rows handled by the specific core.
- **full row vectorization**：The kernel loads the entire row (`offsets = tl.arange(0, N_COLS)`) into registers in a single operation. This maximizes GM→UB access continuity and allows the reduction (`tl.sum`) to be performed using wide vector units, accelerating the variance calculation.
- **precision accumulation**：The input is cast to `float32` (`xf = x.to(tl.float32)`) prior to the squared sum reduction. This ensures numerical stability and prevents overflow/underflow issues that can occur when accumulating squared values in lower precision formats like fp16 or bf16.
- **persistent kernel strategy**：The kernel implements a grid-stride loop using `range(pid, NUM_BLOCKS, NUM_CORES)`. This allows a single program instance to process multiple blocks of rows sequentially, significantly reducing kernel launch overhead and improving load balancing across available compute units.
- **weight broadcasting optimization**：The weight tensor is loaded into registers once before the main loop and broadcasted to the shape `(BLOCK, N_COLS)`. This eliminates redundant global memory accesses for the weight vector inside the loop, which is crucial as the weight is constant across all rows.
- **mixed precision computation**：The input data is cast to `float32` (`xf = x.to(tl.float32)`) prior to the variance and reciprocal square root calculations. This ensures higher numerical stability during the reduction and division operations, preventing potential overflow or underflow, before casting back to the original data type for the final output.
- **two pass algorithm**：The kernel separates the computation into two distinct loops the first pass calculates the variance (reduction), and the second pass applies the normalization and weights. This approach reduces register pressure compared to loading the entire row into local memory, at the cost of increased global memory access (reading X twice).
- **block wise vectorization**：The input data is processed in chunks defined by `blockSize`. This allows the compiler to generate vectorized memory loads/stores, maximizing memory bandwidth utilization and aligning with the hardware memory transaction granularity.
- **precision control**：Intermediate accumulation of the variance is explicitly performed in `tl.float32` (via `.to(tl.float32)`), preventing overflow or underflow that might occur with fp16/bf16 during the summation of squares, before casting back to the output type.
- **jit specialization control**：The `do_not_specialize=["eps"]` argument is used to prevent the JIT compiler from generating a new kernel version for every unique epsilon value, thereby reducing compilation overhead and binary size.
- **dynamic row scheduling**：The kernel dynamically calculates the range of rows each program instance handles based on the total number of programs (grid size). This ensures load balancing even if the total number of rows is not perfectly divisible by the grid size.
- **block wise accumulation**：The reduction of squared values is performed in blocks defined by blockSize. This allows the kernel to handle sequence lengths (n_cols) that exceed the maximum vector load width or register capacity, while maintaining high memory bandwidth utilization through contiguous accesses.
- **precision stabilization**：Intermediate accumulation (sum_sq) and the inverse RMS calculation are explicitly performed in float32. This prevents overflow during the squaring of fp16/bf16 values and ensures sufficient precision for the division operation, which is critical for model convergence.
- **fused weight application**：The normalization and weight multiplication are fused into the second pass over the data. This reduces the total number of memory writes and reads compared to a de-coupled approach (normalize then multiply).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/149-_rms_norm_kernel.py
```

**坏实践（反例）**：逐行重复加载权重向量 + 一行一个 program 的朴素调度
```python
# 每个 program 只处理一行，且行内重复读权重
x = tl.load(X_ptr + row * N_COLS + cols)
w = tl.load(W_ptr + cols)   # 每行重复从 GM 读同一权重向量
```
后果：权重向量随行数被重复从 GM 读取，访存流量线性膨胀；逐行一个 program 还带来过大 grid 与启动开销、负载不均。✅ 正解：权重在主循环外加载一次并广播复用；行维度用 grid-stride 循环（`range(pid, NUM_BLOCKS, NUM_CORES)`）让固定 NUM_CORES 个 program 覆盖所有行。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_001.yaml`)
- 声称性能收益：15%
- 备注：The implementation uses a two-pass strategy over the column dimension (one for reduction, one for application). While this increases memory traffic (loading inputs twice), it is necessary for large hidden sizes to avoid register spilling.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](../compute-pipeline/exp_chunk_cumsum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
