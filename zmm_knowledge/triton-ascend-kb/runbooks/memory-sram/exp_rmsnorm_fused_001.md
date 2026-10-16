---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: 2D vector loading for small hidden dims"
description: "normalization 场景下，Memory Access Optimization: 2D vector loading for small hidden dims，收益 25%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: 2D vector loading for small hidden dims

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，n_cols < 2048 的融合加+归一化前向场景；失效边界：n_cols >= 2048 时整行无法单次驻留 UB，必须改用列 tiling 方案；UB 容量约束：BLOCK_SIZE_M × n_cols × dtype 字节数 × 驻留份数（X、R、S 等）≤ UB 预算，超出需减小 BLOCK_SIZE_M 或改用 tiling）

### 原则
- **2D vector loading**：Utilizes 2D indexing (row_idx[:, None], col_offsets[None, :]) in tl.load to fetch multiple rows (BLOCK_SIZE_M) simultaneously. This maximizes the utilization of the NPU's Unified Buffer (UB) and memory bandwidth, specifically targeting scenarios where n_cols is small (<2048) to avoid tiling overhead.
- **cache policy optimization**：Applies 'evict_first' eviction policy for input tensors (X, R) to retain data in cache for the subsequent reduction step, and uses '.cg' (cache to global) modifier for the intermediate residual output (S) to stream data directly to global memory, reducing cache pollution.
- **instruction level parallelism**：Preloads the weight vector (W_row) outside the main grid-stride loop. This overlaps the memory access of weights with the computation of the residual standard deviation (rstd), effectively hiding memory latency.
- **fused computation pipeline**：Integrates element-wise addition, variance reduction, normalization, and scaling into a single kernel pass. The intermediate sum (S) is computed once and kept in registers to be reused for both the rstd calculation and the final output (Y) generation, significantly reducing global memory access.
- **precision control**：Implements model-specific casting modes to optimize the placement of float32 conversions. This ensures numerical stability for the reduction operation while maintaining high throughput for vector operations by minimizing unnecessary type conversions.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/092-_fused_add_rms_norm_forward_kernel_no_tiling.py
```

**坏实践（反例）**：n_cols < 2048 的小行仍套用列 tiling 分块加载
```python
for col_off in range(0, n_cols, BLOCK_N):   # n_cols=1024 也切块
    x = tl.load(X_ptr + row * n_cols + col_off + offs, ...)
```
后果：循环控制、边界 mask 等 tiling 开销占比升高，小块加载吃不满 GM→UB 连续访存带宽。✅ 正解：整行单次 2D 向量化加载（`row_idx[:, None], col_offsets[None, :]`），权重在 grid-stride 循环外预载，中间和 S 驻留寄存器复用于 rstd 与输出（收益 25%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_fused_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically designed for small hidden dimensions (n_cols < 2048). For larger dimensions, a tiling-based approach is usually preferred to fit data into SRAM/UB.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: small tile strategy](../compute-pipeline/exp_matmul_triton_base_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
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
