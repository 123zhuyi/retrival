---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: micro kernel tiling'
description: 'misc 场景下，Compute Optimization: micro kernel tiling，收益 35%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: micro kernel tiling

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：16×16 tile 是寄存器压力与计算密度的折中（备注自述），更大块需换分治结构；`do_not_specialize=["T"]` 仅在 T 不影响控制流时成立；NTASKS=2 的多任务流水受寄存器预算约束）

### 原则
- **micro kernel tiling**：The computation is aggressively tiled into 16x16 blocks (N_BLOCKS x 16 x 16). This specific tile size is optimized to match the hardware's vector register width and memory transaction size, ensuring high bandwidth utilization and minimizing uncoalesced memory access.
- **matrix accelerated reduction**：The core triangular solve logic replaces traditional scalar loops with vectorized tensor operations. By reshaping data and using `b_a[:, :, None] * b_A` followed by `tl.sum`, the reduction is mapped to the hardware's matrix multiplication or vector dot-product units, significantly accelerating the compute-bound forward substitution phase.
- **dynamic sequence handling**：The kernel efficiently handles variable-length sequences (ragged tensors) using `cu_seqlens` and `chunk_indices`, eliminating the need for padding and wasted computation. The `do_not_specialize=["T"]` attribute ensures the compiled kernel is reusable across different sequence lengths, reducing JIT compilation overhead.
- **task pipelining**：The implementation uses a `NTASKS` loop (set to 2) to process multiple chunks of data per program ID. This increases instruction-level parallelism and helps hide memory latency by keeping the compute pipeline busy.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/153-solve_tril_16x16_kernel.py
```

**坏实践（反例）**：前代（forward substitution）用标量循环逐元素求解 → 计算 Bound 阶段完全绕开矩阵/向量单元 → ✅ 正解即本条：16×16 微块 + reshape 后用 `b_a[:,:,None] * b_A` + `tl.sum` 把归约映射到向量/矩阵单元 + cu_seqlens 处理变长。

**已知实例**：misc §raw (`experiences/misc/exp_tril_solve_001.yaml`)
- 声称性能收益：35%
- 备注：This optimization is particularly effective for attention mechanisms or linear algebra operations requiring lower-triangular matrix inversion/solving on NPU/GPU hardware. The 16x16 block size is a trade-off between register pressure and arithmetic intensity.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](../portability/exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](../portability/exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc

<!-- okf:related:end -->
