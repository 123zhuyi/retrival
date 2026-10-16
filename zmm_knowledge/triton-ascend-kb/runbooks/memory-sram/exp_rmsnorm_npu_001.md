---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: column blocking for UB management"
description: "normalization 场景下，Memory Access Optimization: column blocking for UB management，收益 35%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: column blocking for UB management

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，n_cols >= 256 的融合加+归一化场景；失效边界：n_cols 小到整行可单次驻留 UB 时，列分块引入的循环开销反而拖累，宜用非 tiling 整行方案；UB 容量约束：BLOCK_SIZE × dtype 字节数 × 驻留份数（X、R、S）≤ UB 预算；两遍法需将中间 S 写回再读（.cg 缓存兜底），GM 带宽极度饱和时该额外流量会显现）

### 原则
- **grid stride scheduling**：Implements a grid-stride loop using `tl.range(pid, n_rows, num_progs, num_stages=NUM_STAGES)`. This allows the kernel to process an arbitrary number of rows (`n_rows`) efficiently regardless of the physical NPU core count, ensuring high utilization by mapping multiple rows to a single program instance.
- **column blocking ub management**：Processes wide rows in chunks defined by `blockSize` rather than loading the entire row at once. This tiling strategy is critical for preventing Unified Buffer (UB) overflow on NPUs when `n_cols` is large, a common failure point in naive implementations.
- **two pass reduction algorithm**：Decouples the computation into two distinct phases. The first pass computes the residual `S = X + R` and accumulates the `sum_square` for the RMS statistic. The second pass reloads `S` and applies the normalization and weights. This separation is necessary to calculate the global mean square statistic before applying the normalization.
- **explicit cache policy**：Utilizes specific cache modifiers to optimize data movement. `eviction_policy="evict_first"` is used for inputs `X` and `R` to stream data without polluting the cache, while `cache_modifier=".cg"` is used when storing the intermediate `S` to ensure it remains in the cache for the subsequent load in the second pass, significantly reducing memory latency.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/093-_fused_add_rms_norm_forward_kernel_npu.py
```

**坏实践（反例）**：naive 整行一次加载，不做列分块
```python
x_row = tl.load(X_ptr + row * n_cols + tl.arange(0, n_cols))  # n_cols 很大
r_row = tl.load(R_ptr + row * n_cols + tl.arange(0, n_cols))
```
后果：整行 X、R 及中间量超出 UB 容量，编译期资源分配失败或运行期 spill——naive 实现的常见失败点。✅ 正解：按 blockSize 列分块 + 两遍算法（第一遍算 S=X+R 与 sum_square，第二遍重载 S 做归一化），配 `evict_first`/`.cg` cache 策略（收益 35%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_npu_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically optimized for scenarios where the hidden dimension (n_cols) is large, exceeding the capacity of the NPU's Unified Buffer if loaded in a single block. 支持多种模型特定的精度转换（cast）模式。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
