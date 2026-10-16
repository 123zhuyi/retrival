---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: no-tiling single-load for small rows"
description: "normalization 场景下，Memory Access Optimization: no-tiling single-load for small rows，收益 20%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: no-tiling single-load for small rows

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，n_cols < 2048 的归一化前向场景；失效边界：n_cols >= 2048 时整行超出 UB 容量，必须改用列 tiling 方案；UB 容量约束：n_cols × dtype 字节数 × 驻留份数（两遍加载 + fp32 升精度中间量）≤ UB 预算，逼近上限时应减小驻留份数或切换 tiling）

### 原则
- **no tiling strategy**：The kernel loads the entire row vector in a single operation without column chunking. This is specifically optimized for small hidden dimensions (n_cols < 2048) to eliminate the overhead of tiling logic, loop control, and boundary checks associated with blocked algorithms.
- **grid stride scheduling**：Utilizes a grid-stride loop over the row dimension (`tl.range(pid, n_rows, num_progs)`). This allows a fixed number of program instances (tuned to NPU core count) to process an arbitrary number of rows efficiently, maximizing hardware utilization without launching excessive grid blocks.
- **cache hierarchy optimization**：Implements a two-pass loading strategy with specific cache policies. The first load (for variance calculation) uses `eviction_policy="evict_first"` to stream data without polluting the cache, while the second load (for normalization) uses `cache_modifier=".ca"` to cache data for the subsequent element-wise affine transformations.
- **precision control pipeline**：Supports model-specific casting modes by upcasting inputs to FP32 during the reduction and normalization phases to ensure numerical stability, followed by a downcast back to the original data type before storage, balancing accuracy and performance.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/100-_rms_norm_forward_kernel_no_tiling.py
```

**坏实践（反例）**：小列场景（n_cols < 2048）仍引入列分块循环
```python
for off in range(0, n_cols, BLOCK):      # n_cols=512 也切块
    x = tl.load(X_ptr + row * n_cols + off + offs, ...)
```
后果：tiling 的循环控制、边界检查开销在小行上占比升高，多次小块加载吃不满 GM→UB 连续访存带宽。✅ 正解：整行单次加载消除 tiling 开销；两遍 cache 策略分工——首遍 `evict_first` 流式读不污染缓存，二遍 `.ca` 缓存复用做仿射变换（收益 20%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_notiling_001.yaml`)
- 声称性能收益：20%
- 备注：This kernel is designed specifically for scenarios where the column dimension is small (< 2048). For larger dimensions, a tiled approach is preferred to fit within memory constraints and maximize parallelism.

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
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
