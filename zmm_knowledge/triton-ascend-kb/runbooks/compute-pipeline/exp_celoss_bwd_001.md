---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: two pass algorithm'
description: 'loss 场景下，Compute Optimization: two pass algorithm，收益 35%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, loss]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: two pass algorithm

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：两遍策略以重读输入换寄存器/片上容量，C 或 D < 128 时中间矩阵本可驻留，重读开销无收益；BLOCK_C/BLOCK_D 需按目标硬件 SRAM 容量调优；do_not_specialize 不适用于影响控制流的标量）

### 原则
- **two pass algorithm**：The kernel separates the reduction phase (calculating max and sum of exponents) from the gradient accumulation phase. This avoids storing the large intermediate probability matrix (C x D) in global memory, significantly reducing memory bandwidth pressure and memory footprint, at the cost of re-reading the input tensor.
- **numerical stability**：Implements the 'max trick' (subtracting max value before exp) during the reduction loop to prevent floating-point overflow/underflow, which is critical for the stability of Softmax calculations in fp16/bf16.
- **block tiling and autotuning**：Utilizes `BLOCK_C` and `BLOCK_D` tiling to process data chunks that fit into the SRAM/Cache. The `@triton.autotune` decorator dynamically selects the optimal block sizes based on the specific C and D dimensions to maximize hardware utilization.
- **vectorized masking**：Efficiently handles boundary conditions and the `ignore_index` logic using vectorized mask operations (`tl.where`, `ignore_mask`) rather than scalar branching, maintaining high pipeline throughput.
- **stable log sum exp reduction**：The first loop implements a numerically stable reduction algorithm for the Softmax denominator. It iteratively updates `tmp_max` and `tmp_sum` using the `max` and `exp` shifting technique (`tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp`). This prevents floating-point overflow/underflow when computing exponents of large logits, which is critical for large vocabulary sizes (C).
- **two pass gradient computation**：The kernel employs a two-pass strategy over the class dimension `C`. The first pass aggregates global statistics (sum of exponentials and weighted target sum), and the second pass computes the per-element gradient. This trades off increased memory traffic (reloading inputs) for significantly reduced register/Shared Memory pressure, allowing the kernel to handle large batch sizes or hidden dimensions that would otherwise spill registers.
- **fused label smoothing and weighting**：The logic for label smoothing (`tgt * (1 - label_smoothing) + label_smoothing / C`) and sample weighting (`w`) is fused directly into the kernel loops. This eliminates the need for pre-processing the target tensor or launching separate auxiliary kernels, reducing memory footprint and kernel launch overhead.
- **vectorized tiled memory access**：The kernel utilizes 2D block tiling (`BLOCK_C` x `BLOCK_D`) to load data. It uses `tl.load` with masks to efficiently handle non-divisible dimensions, ensuring high memory bandwidth utilization while correctly processing boundary elements without padding artifacts.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/112-celoss_probability_bwd.py
```

**坏实践（反例）**：把 C×D 的 softmax 概率矩阵整体物化到 HBM 再算梯度 → 大词表（C≥128×大 batch）下显存容量与带宽双双成为瓶颈 → ✅ 正解即本条：两遍策略——第一遍只归约 max/sum（max-trick 防溢出），第二遍算梯度，中间概率矩阵不落 HBM。

**已知实例**：loss §raw (`experiences/loss/exp_celoss_bwd_001.yaml`)
- 声称性能收益：35%
- 备注：This implementation prioritizes numerical stability and memory capacity over minimizing global memory loads (due to the two-pass approach). It is best suited for scenarios where C (number of classes) is large, preventing the storage of the full [N, C, D] intermediate activation matrix in SRAM.

<!-- okf:related:start -->

# 相关

- 开发指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 loss
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](../memory-sram/exp_gn_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: dynamic workload tiling](../memory-sram/exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: grid stride scheduling](../memory-sram/exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](../memory-sram/exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](../memory-sram/exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](../memory-sram/exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](../memory-sram/exp_rmsnorm_grad_dx_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](../memory-sram/exp_rmsnorm_infer_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](../memory-sram/exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column blocking for UB management](../memory-sram/exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column tiling for wide rows](../memory-sram/exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block size tuning](../memory-sram/exp_sum_scale_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
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
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
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
