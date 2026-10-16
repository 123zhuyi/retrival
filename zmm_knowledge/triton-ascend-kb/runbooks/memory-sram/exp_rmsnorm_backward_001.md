---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation"
description: "normalization 场景下，小列宽（n_cols<2048）反向的整行向量化加载与无冲突 dW 累加，收益 25%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，n_cols <2048；失效边界：n_cols ≥2048 时整行驻留超过 UB/寄存器预算，应改用列分块版本；dW 输出为按 pid 分区的部分和，需二次归约 kernel 聚合，n_rows 很小时该额外 kernel 的开销占比升高）

### 原则
- **register reuse**：Reuses the computed X_normalized (X * rstd) for both the dX and dW gradient calculations, minimizing redundant multiplication operations and reducing register pressure.
- **casting optimization**：Implements 模型特定的精度转换模式 to handle mixed-precision arithmetic efficiently, converting to fp32 only when necessary for accumulation to maintain numerical stability.
- **vectorized loading**：Utilizes 2D vector loading strategies tailored for small column sizes (e.g., 1x2048) to maximize Unified Buffer (UB) utilization and memory bandwidth efficiency.
- **grid stride loop**：Uses a grid-stride loop over the row dimension to handle variable batch sizes efficiently while keeping the block size fixed to manage register pressure.
- **local accumulation**：Accumulates dW gradients in a per-program buffer (dW_ptr + pid) to avoid atomic operations and reduce global memory contention during the backward pass.
- **row wise vectorization**：The kernel abandons column tiling in favor of loading the entire row (up to blockSize) in a single vectorized load. This significantly reduces loop control overhead and maximizes memory bandwidth utilization for tensors with small hidden dimensions (n_cols < 2048).
- **grid stride scheduling**：Utilizes a grid-stride loop (`tl.range` with `num_stages`) to allow each program to process multiple rows. This increases instruction-level parallelism and better hides memory access latency by overlapping the loads of subsequent rows with the computation of the current row.
- **cache eviction policy**：Explicitly sets `eviction_policy="evict_first"` for loading input tensors (dY, X). Since these rows are read-only once per iteration, this policy prevents them from polluting the cache, ensuring that cache capacity is preserved for frequently accessed data like weights (W) or reducing thrashing.
- **conflict free dw accumulation**：Implements a partial reduction strategy for the weight gradient (dW) by writing results to a buffer indexed by the program ID (`pid`) rather than using atomic adds to a global address. This eliminates serialization and contention caused by atomic operations, significantly improving throughput in the backward pass.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/102-_rms_norm_backward_kernel_no_tiling.py
```

**坏实践（反例）**：dW 梯度用标量 `tl.atomic_add` 直接累加到全局地址
```python
for i in range(BLOCK):
    tl.atomic_add(dW_ptr + i, dw_partial[i])   # 各 program 竞争同一全局地址
```
后果：原子操作串行化，program 数越多竞争越激烈，反向吞吐被写冲突拖垮。✅ 正解：dW 写入按 pid 分区的部分和 buffer（`dW_ptr + pid * n_cols`），无原子无冲突，事后用一个二次归约 kernel 跨 program 聚合。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_backward_001.yaml`)
- 声称性能收益：25%
- 备注：This optimization is specifically tailored for small n_cols (< 2048). The dW output produced by this kernel contains partial sums and requires a secondary reduction kernel to aggregate results across the program dimension.

<!-- okf:related:start -->

# 相关

- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
