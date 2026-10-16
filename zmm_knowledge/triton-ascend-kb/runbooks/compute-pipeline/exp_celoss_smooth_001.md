---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: numerical stability'
description: 'loss 场景下，Compute Optimization: numerical stability，收益 30%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, loss]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: numerical stability

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：C < 1024 时输入重读开销占比升高、两遍策略收益下降；label_smoothing=0 且无 ignore_index 时融合的 smoothing 分支是死代码，应由编译期开关裁剪；输入本身高精度（fp32）时 LogSumExp trick 的紧迫性低于 fp16/bf16）

### 原则
- **numerical stability**：Implements the Log-Sum-Exp trick (calculating max, then exp(x-max), then log(sum) + max) to prevent overflow/underflow when computing softmax probabilities in low precision (fp16/bf16).
- **two pass algorithm**：Separates the computation into two passes over the class dimension. The first pass calculates the log-partition function (normalization term), and the second pass computes the weighted loss. This avoids redundant expensive exponential and logarithmic calculations during the loss accumulation.
- **vectorized batch processing**：Utilizes BLOCK_D to process multiple samples (or sequence tokens) in parallel within a single kernel thread. This maximizes memory bandwidth utilization and instruction throughput by amortizing the loop overhead over the batch dimension.
- **efficient smoothing logic**：Uses vectorized `tl.where` operations to apply label smoothing factors based on target indices, avoiding scalar loops or complex gather operations for the smoothing distribution.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/110-celoss_indices_smooth_kernel.py
```

**坏实践（反例）**：先单独跑一个 kernel 把 label smoothing 后的目标分布物化到 HBM，再算 loss → 多出一整个 [.., C] 张量的写+读，大词表下带宽开销显著 → ✅ 正解即本条：用 `tl.where` 向量化地在主循环内直接应用 smoothing 因子，两遍结构消除重复 exp/log 计算。

**已知实例**：loss §raw (`experiences/loss/exp_celoss_smooth_001.yaml`)
- 声称性能收益：30%
- 备注：This kernel reads the input tensor twice (once for max/sum, once for loss). It is optimized for scenarios where C (number of classes) is large (e.g., vocabulary size in LLMs).

<!-- okf:related:start -->

# 相关

- 开发指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 loss
- 相关主题: [Memory Access Optimization: dynamic workload tiling](../memory-sram/exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: grid stride scheduling](../memory-sram/exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](../memory-sram/exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](../memory-sram/exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](../memory-sram/exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column blocking for UB management](../memory-sram/exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column tiling for wide rows](../memory-sram/exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: block size tuning](../memory-sram/exp_sum_scale_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
