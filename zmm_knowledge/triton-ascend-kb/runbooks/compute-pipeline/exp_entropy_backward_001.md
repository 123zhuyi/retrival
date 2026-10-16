---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: kernel fusion and recomputation'
description: 'loss 场景下，Scheduling Optimization: kernel fusion and recomputation，收益 25%'
tags: [多核并行与负载均衡, matrix-operations, loss]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: kernel fusion and recomputation

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：vocab_size < 128 时重算 logits 的 tl.dot 开销可能高于直接 load；原子加假设 pid_m 分布稀疏、竞争低，索引集中时收益下降；GROUP_SIZE_M=16 等配置绑定 910 系列，换硬件需重调）

### 原则
- **kernel fusion and recomputation**：The kernel fuses the matrix multiplication (logits recomputation) with the backward pass of softmax and entropy loss. By recomputing the logits on-the-fly via `tl.dot` instead of loading them from global memory, it significantly reduces memory bandwidth consumption, which is critical for operations involving large vocabulary sizes.
- **hardware specific type optimization**：Implements a specific optimization for Ascend NPU by casting index operands to `int32` before performing equality checks for mask generation (`mask = (offs_bn + rank * vocab_size)[None, :].to(tl.int32) == labels[:, None]`). This avoids the performance degradation associated with `int64` scalar operations on the NPU.
- **grouped scheduling for locality**：Utilizes `GROUP_SIZE_M` (set to 16) to ensure that program IDs processing consecutive rows of the matrix are scheduled within the same group. This improves cache locality for the `hidden` input tensor and the reduction scalars (`maximum`, `accu`), reducing redundant global memory accesses.
- **block tiling strategy**：Employs a tiling configuration of `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, and `BLOCK_SIZE_K=128`. This configuration is tuned to balance register pressure and memory throughput, ensuring that the Tensor Core units are efficiently utilized during the matrix multiplication phase.
- **grid stride scheduling**：The kernel utilizes a grid-stride loop (`for pid in range(pid0, total_tiles, num_aicores)`) to map logical tiles to physical AI cores. This significantly reduces kernel launch overhead and scheduling latency, particularly beneficial when the total number of tiles (vocab_size * num_tokens) is much larger than the number of available physical cores.
- **grouped tiling strategy**：Implements a `GROUP_SIZE_M` parameter to process multiple rows of the M dimension (tokens) together for a given range of the N dimension (vocab). This enhances data locality and reuse of the weight matrix loaded from global memory, reducing memory bandwidth pressure.
- **compiler friendly pointer arithmetic**：Inside the accumulation loop, pointer addresses are calculated explicitly using `base + offset + k * stride` rather than incrementing a pointer variable. This specific pattern assists the compiler in performing alias analysis and inferring contiguous memory access patterns, often leading to better vectorization and load/store efficiency.
- **in kernel atomic reduction**：The gradient accumulation over the K dimension is handled within the kernel using `tl.atomic_add`. This avoids the need for a separate reduction kernel or intermediate global memory buffers, lowering overall memory footprint and latency.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/208-efficient_entropy_backward_kernel_d_hidden.py
```

**坏实践（反例）**：反向时把大词表 logits 从 HBM 重新 load 进来 → [num_tokens, vocab_size] 级带宽消耗压垮内存子系统 → ✅ 正解即本条：用 `tl.dot` 就地重算 logits（重算换带宽），索引用 int32 比较避免 NPU 上 int64 标量运算劣化。

**已知实例**：loss §raw (`experiences/loss/exp_entropy_backward_001.yaml`)
- 声称性能收益：25%
- 备注：This optimization is specifically designed for the backward pass of entropy loss with large vocabulary sizes. The atomic add strategy assumes that the hardware handles atomic contention efficiently or that contention is low due to the distribution of `pid_m`.

<!-- okf:related:start -->

# 相关

- 开发指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 loss
- 相关主题: [Memory Access Optimization: vectorized offset generation](../memory-sram/exp_attn_causal_mask_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: tiling strategy](../memory-sram/exp_flash_attn_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_hstu_attn_fwd_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](../memory-sram/exp_paged_attn_flash_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: paged kv indirection](../memory-sram/exp_paged_prefill_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 主题相关：共享 loss, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 loss, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 loss, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 loss, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
