---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: persistent kernel grid stride loop'
description: 'misc 场景下，Scheduling Optimization: persistent kernel grid stride loop，收益 25%'
tags: [多核并行与负载均衡, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: persistent kernel grid stride loop

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：total_blocks 不超过物理核数时 persistent 模式无摊薄收益；GRID_SIZE 需按核数设定，过大/过小都损失 occupancy；片上 `tl.trans` 转置有额外开销，B 若已是目标 layout 则不需要）

### 原则
- **persistent kernel grid stride loop**：The kernel utilizes a fixed-size 1D grid (`GRID_SIZE`) where each program iterates over multiple output blocks via `for block_index in range(pid, total_blocks, GRID_SIZE)`. This strategy significantly reduces kernel launch overhead and maintains high hardware occupancy by keeping compute units active throughout the processing of multiple tiles, rather than launching a new program instance for every single tile.
- **explicit software transpose**：Matrix B is loaded with shape [blockN, blockK] and explicitly transposed to [blockK, blockN] using `tl.trans` before the `tl.dot` operation. This allows the kernel to handle B's memory layout flexibly (potentially optimizing for cache locality during the load phase) while satisfying the input requirements of the matrix multiplication unit.
- **boundary safe tiling**：Implements robust boundary handling using dynamic masks (`m_mask`, `n_mask`, `k_mask`) derived from `tl.arange`. This ensures correct computation and memory access for matrices where dimensions (M, N, K) are not integer multiples of the block sizes (blockM, blockN, blockK), preventing out-of-bounds errors without requiring input padding.

### 代码/骨架出处
- `TritonPath`:
```python
linear_persistent_kernel
```

**坏实践（反例）**：每个输出 tile 发射一个 program（1:1 grid）→ tile 数远超核数时 launch/调度开销主导；维度不整除时直接越界或要求上游 padding → ✅ 正解即本条：固定 GRID_SIZE 的 persistent grid-stride + m/n/k 三路动态 mask 边界处理。

**已知实例**：misc §raw (`experiences/misc/exp_linear_persistent_001.yaml`)
- 声称性能收益：25%
- 备注：This persistent kernel pattern is highly effective for large-scale matrix multiplications where the number of output tiles exceeds the number of available physical cores, minimizing the latency associated with kernel scheduling and launch.

<!-- okf:related:start -->

# 相关

- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — serial persistent 骨架对应 persistent kernel grid-stride 模式
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: persistent kernel strategy](../memory-sram/exp_layernorm_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](../portability/exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](../portability/exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
