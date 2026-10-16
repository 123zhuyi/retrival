---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: two pass algorithm'
description: 'loss 场景下，Compute Optimization: two pass algorithm，收益 30%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, loss]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: two pass algorithm

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：C < 128 时单块可容纳整个分类维，两遍策略的重读是纯开销；fp32 输入时 max-trick 仍建议保留但紧迫性低于 fp16/bf16；do_not_specialize 仅适用于不影响控制流的标量）

### 原则
- **two pass algorithm**：The kernel implements a two-pass reduction strategy. The first loop iterates over the class dimension (C) to compute the global maximum (log-sum-exp trick) and the normalization denominator (sum of weights). The second loop computes the actual gradients. This avoids materializing the large intermediate Softmax probability tensor, significantly reducing memory bandwidth pressure.
- **numerical stability**：Utilizes the "online softmax" algorithm within the reduction loop. By incrementally updating the running maximum and sum (`cur_max`, `tmp_sum * tl.exp(tmp_max - cur_max)`), it ensures numerical stability for floating-point accumulation, which is critical when using fp16 or bf16 data types.
- **vectorized tiling**：Employs 2D block tiling (`BLOCK_C` x `BLOCK_D`) to process the logits and targets. This maximizes global memory throughput by ensuring loads and stores are coalesced, and allows efficient utilization of the SRAM/Cache.
- **logic fusion**：Fuses the label smoothing calculation and the `ignore_index` masking directly into the gradient computation loop. This eliminates the need for separate preprocessing kernels or extra memory reads for the smoothing factors.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/113-celoss_indices_smooth_bwd.py
```

**坏实践（反例）**：反向时把完整 Softmax 概率矩阵写回 HBM 再读回算梯度 → 大词表下 [N, C, D] 中间激活超出片上容量，带宽成为瓶颈 → ✅ 正解即本条：online max/sum 归约 + 两遍梯度计算 + smoothing/ignore_index 融合进循环。

**已知实例**：loss §raw (`experiences/loss/exp_celoss_smooth_bwd_001.yaml`)
- 声称性能收益：30%
- 备注：This configuration is optimized for large vocabulary sizes (C) where storing the full Softmax matrix is prohibitive. The `do_not_specialize` attribute on scalar arguments helps reduce compilation time for dynamic shapes.

<!-- okf:related:start -->

# 相关

- 开发指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 loss
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](../memory-sram/exp_gn_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: dynamic workload tiling](../memory-sram/exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: grid stride scheduling](../memory-sram/exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](../memory-sram/exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](../memory-sram/exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](../memory-sram/exp_rmsnorm_grad_dx_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](../memory-sram/exp_rmsnorm_infer_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: column blocking for UB management](../memory-sram/exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column tiling for wide rows](../memory-sram/exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
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
