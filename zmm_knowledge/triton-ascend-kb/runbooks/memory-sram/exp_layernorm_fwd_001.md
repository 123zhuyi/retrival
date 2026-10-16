---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: persistent kernel strategy"
description: "normalization 场景下，Memory Access Optimization: persistent kernel strategy，收益 20%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: persistent kernel strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，n_rows ≥1、n_cols ≥128；失效边界：带宽敏感且单遍可完成统计的场景，三遍读 X 的带宽开销会稀释收益；n_rows 小于可用核数时 persistent 循环无法摊薄 launch 开销）

### 原则
- **persistent kernel strategy**：The kernel implements a persistent loop structure (`for i in tl.range(n_iters)`) where a single program instance processes multiple rows (`BLOCK_ROWS`) based on the number of available cores (`N_CORES`). This reduces kernel launch overhead and improves hardware utilization by keeping the compute units busy, particularly beneficial for large batch sizes or sequence lengths.
- **multi operator fusion**：The kernel fuses multiple stages of the normalization forward—including mean/variance calculation, normalization, affine transformation (weight/bias), residual connection (Z), and gating activation—into a single kernel. This drastically reduces global memory read/write traffic by avoiding the storage of intermediate tensors.
- **compile time specialization**：By utilizing `tl.constexpr` for feature flags such as `HAS_BIAS`, `HAS_Z`, `IS_RMS_NORM`, and `NORM_BEFORE_GATE`, the compiler generates specialized code paths for different configurations. This eliminates runtime branch divergence and ensures optimal instruction pipelining.
- **precision management**：Intermediate calculations for mean and variance are explicitly cast to `tl.float32` to maintain numerical stability and prevent overflow/underflow, which is critical for training convergence in mixed-precision scenarios.
- **block tiling strategy**：The kernel utilizes a 2D tiling strategy defined by BLOCK_SIZE_M and BLOCK_SIZE_N to load input data X into SRAM. This reduces global memory access frequency compared to scalar operations and improves data reuse within the block.
- **multi pass algorithm**：The calculation is split into three distinct passes over the data one for calculating the Mean, one for calculating the Variance (RSTD), and a final pass for computing the Output. This approach significantly reduces register pressure per pass compared to a single-pass fused implementation, at the cost of increased global memory bandwidth usage (reading X three times).
- **fp32 accumulation**：Intermediate accumulations for sum (mean) and sum of squares (variance) are explicitly cast to and computed in float32. This ensures numerical stability and prevents overflow/underflow, which is critical when inputs are fp16 or bf16.
- **heuristic scheduling**：The BLOCK_SIZE_M is determined via @triton.heuristics, allowing the kernel to dynamically adapt the tile size based on the input matrix dimensions (n_rows) to optimize hardware occupancy and resource utilization.

### 代码/骨架出处
- `TritonPath`:
```python
_layernorm_fwd_kernel
```

**坏实践（反例）**：多遍算法中无消费的重读——每个 pass 完整重读输入，且在不使用权重/偏置的 pass 里重复加载
```python
for pass_id in range(3):
    x = tl.load(x_ptr + offs)     # 均值/方差/输出三遍都从 GM 完整重读 X
    w = tl.load(w_ptr)            # 方差 pass 内加载但计算中未使用（冗余）
    b = tl.load(b_ptr)
```
后果：GM 带宽浪费在无消费的重读上，bandwidth-bound 场景成为瓶颈。✅ 正解：统计量能单遍/两遍完成时改用融合归约（如 Welford 合并均值与方差），减少 X 的 GM 读次数；权重/偏置只在真正消费的 pass 加载，未使用的 load 直接删除。

**已知实例**：normalization §raw (`experiences/normalization/exp_layernorm_fwd_001.yaml`)
- 声称性能收益：20%
- 备注：当前实现从 GM 读三次输入 X，对带宽敏感场景非最优；方差循环中存在一次未参与计算的权重/偏置冗余 load，应删除。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](../compute-pipeline/exp_entropy_backward_d_weight_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](../compute-pipeline/exp_linear_persistent_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](../compute-pipeline/exp_partial_matmul_dynamic_001.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](../compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
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
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
