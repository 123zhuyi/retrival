---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: diagonal traversal pattern'
description: 'misc 场景下，Scheduling Optimization: diagonal traversal pattern，收益 20%'
tags: [多核并行与负载均衡, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: diagonal traversal pattern

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：仅当 num_pid_m 与 num_pid_n 都 ≥ BLOCK_TRESHHOLD（大网格）才启用，小网格走默认行优先；收益来自 memory bank 争用缓解，带宽不是瓶颈的 kernel 无收益；LCM 步长公式依赖阈值维度的具体取值，需按硬件 bank 结构调）

### 原则
- **diagonal traversal pattern**：Replaces standard row-major PID mapping with a diagonal traversal strategy. By calculating `task_m_idx` and `task_n_idx` such that blocks are processed along diagonals, this distributes memory access requests more evenly across memory banks. This significantly reduces contention and alleviates bandwidth bottlenecks common in large-scale matrix multiplications.
- **conditional activation**：The optimization is gated by `BLOCK_TRESHHOLD`. The complex diagonal logic is only engaged when both grid dimensions (`num_pid_m`, `num_pid_n`) are sufficiently large. For smaller grids, the function defaults to simple row-major mapping (`pid // num_pid_n`, `pid % num_pid_n`) to avoid unnecessary computational overhead for index calculation.
- **lcm based stride**：Utilizes the Least Common Multiple (LCM) of the current threshold dimensions (`curThresholdM`, `curThresholdN`) to compute the stride for `task_n_idx`. The formula `(localRelativeBlock + (localRelativeBlock // lcm))` ensures that the access pattern skips ahead optimally, maximizing cache line efficiency and minimizing repeated access to the same memory resources within the active block window.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/239-grouped_launch_diagonal.py
```

**坏实践（反例）**：大网格 GEMM 用默认行优先 pid 映射（`pid // num_pid_n`, `pid % num_pid_n`）→ 相邻 program 集中访问同一 memory bank/相同输入切片，带宽瓶颈被放大 → ✅ 正解即本条：对角遍历 + LCM 步长打散访存分布，小网格自动回退行优先。

**已知实例**：misc §raw (`experiences/misc/exp_launch_diagonal_001.yaml`)
- 声称性能收益：20%
- 备注：This function is a grid mapping utility intended to be called inside the kernel's PID calculation logic. It is particularly effective for GEMM operations where the M and N dimensions are large and memory bandwidth is the limiting factor.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](../portability/exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](../portability/exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
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
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
