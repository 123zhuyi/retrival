---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: column tiling for wide rows"
description: "normalization 场景下，Memory Access Optimization: column tiling for wide rows，收益 25%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: column tiling for wide rows

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，n_cols >= 2048 的归一化前向场景；失效边界：n_cols < 2048 时每行双载（归约遍 + 归一化遍）的额外 GM 读流量无法被摊销，宜改用非 tiling 整行方案；UB 容量约束：BLOCK_SIZE × dtype 字节数 × 驻留份数 ≤ UB 预算；GM 带宽已饱和的场景需权衡双载开销）

### 原则
- **column tiling strategy**：The kernel tiles the column dimension (n_cols) into fixed-size blocks (blockSize). This prevents Unified Buffer (UB) overflow when processing rows with large dimensions (e.g., >= 4096), which is a common bottleneck in NPU memory architecture.
- **grid stride loop**：Implements a grid-stride loop over rows (tl.range(pid, n_rows, num_progs)). This optimizes hardware utilization by matching the number of active program instances to the NPU core count, reducing launch overhead and ensuring efficient scheduling for varying batch sizes.
- **two pass algorithm**：Decouples the reduction phase (sum of squares calculation) from the normalization phase. This allows the kernel to compute the global statistics (rstd) required for normalization while processing data in manageable chunks.
- **cache policy optimization**：Uses 'evict_first' during the reduction pass to minimize cache pollution for data that is read once, and '.ca' (cache all) during the normalization pass to maximize data reuse for weights and inputs.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/101-_rms_norm_forward_kernel_tiled.py
```

**坏实践（反例）**：大列（n_cols >= 4096）仍整行单块加载
```python
row = tl.load(x_ptr + row_off + tl.arange(0, n_cols))  # 整行进 UB
ss = tl.sum(row * row)
```
后果：整行超出 UB 容量，标准 kernel OOM 或 UB 溢出。✅ 正解：列维切 blockSize 块 + grid-stride 行循环匹配核数 + 两遍算法（归约遍 `evict_first` 防污染、归一化遍 `.ca` 促复用），双载开销被大行处理能力摊销（收益 25%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_tiled_001.yaml`)
- 声称性能收益：25%
- 备注：Specifically designed for scenarios with large hidden dimensions (n_cols >= 2048) where standard kernels might OOM or overflow UB. Involves a double-load overhead per row which is amortized by the ability to process large rows.

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
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
