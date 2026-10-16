---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: dual task pipelining'
description: 'attention 场景下，Scheduling Optimization: dual task pipelining，收益 55%'
tags: [多核并行与负载均衡, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: dual task pipelining

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：依赖 Ascend DL 扩展 API（async_task / cross_flag），纯 Triton 语义环境不可用；短序列（N_CTX < 128）时双任务与同步开销无法摊薄；多 workspace 缓冲占用 L1/L0，HEAD_DIM 过大时缓冲溢出即失效）

### 原则
- **dual task pipelining**：The kernel implements a dual-task scheduling strategy where a single program ID (pid) processes two independent sequence blocks (block_idx_0 and block_idx_1) in a loop. This effectively doubles the workload per core launch, overlapping memory access latency of one task with the computation of the other.
- **cube vector asynchronous execution**：The code explicitly separates the compute graph into `dl.async_task(scope=dl.async_task.cube)` for matrix multiplications (score 矩阵乘 inputA·inputB^T 和 Softmax·inputC) and `dl.async_task(scope=dl.async_task.vector)` for element-wise operations (Softmax, masking). This allows the NPU's Cube and Vector units to operate in parallel rather than sequentially.
- **hardware synchronization mechanism**：Utilizes `dl.wait_cross_flag` and `dl.set_cross_flag` to implement a fine-grained producer-consumer synchronization model between the Cube and Vector units. This ensures that the Vector unit starts processing a tile immediately after the Cube unit writes it to the workspace, minimizing pipeline stalls.
- **multi level workspace buffering**：Leverages multiple workspace buffers (WS1, WS2, WS3, WS4) to store intermediate results (中间 score 矩阵与部分输出) in high-bandwidth memory (L1/L0). This avoids spilling intermediate data to global memory (HBM), significantly reducing memory bandwidth pressure.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/055-_attn_fwd_split_cv.py
```

**坏实践（反例）**：一个 pid 只处理一个序列块，且 Cube 单元算完矩阵乘后再串行启动 Vector 单元做 softmax/mask → Cube 与 Vector 交替空转，长序列下近一半算力闲置，kernel launch 次数翻倍 → ✅ 正解即本条：单 pid 双任务 + cube/vector async_task 分离 + cross_flag 生产者-消费者同步。

**已知实例**：attention §raw (`experiences/attention/exp_attn_ascend_dual_pipeline_001.yaml`)
- 声称性能收益：55%
- 备注：This optimization relies on the specific DL (Device Link) API extensions for Ascend hardware. It is highly effective for long sequence lengths (N_CTX) where the overhead of kernel launch is amortized over the large compute volume.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: vectorized offset generation](../memory-sram/exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: block pointer abstraction](../memory-sram/exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](../memory-sram/exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: tiling strategy](../memory-sram/exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: indirect load scalarization](../memory-sram/exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](../memory-sram/exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: paged kv indirection](../memory-sram/exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](../memory-sram/exp_rope_forward_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](../memory-sram/exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: block ptr abstraction](../memory-sram/exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: sparse iteration logic](../memory-sram/exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_tv_distance_001.md) — 同属优化知识，主题相关
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](../portability/exp_attn_bwd_q_down_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](../portability/exp_attn_sda_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](../portability/exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](../portability/exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](../portability/exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 attention, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
