---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: operator fusion"
description: "attention 场景下，Memory Access Optimization: operator fusion，收益 35%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: operator fusion

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，M/N/K ≥64；失效边界：序列很短、中间评分矩阵 M×N 本可留在片上时，融合收益稀释；UB 容量不足以同时容纳常驻输入块与累加器时需先下调 BLOCK_M/BLOCK_N，极端时无法单 kernel 承载）

### 原则
- **operator fusion**：Fuses the score dot product, SiLU activation, normalization, and final weighted value dot product into a single kernel. This prevents the intermediate score matrix (M x N) from being written back to global memory (GM), drastically reducing memory bandwidth consumption.
- **activation inlining**：The SiLU activation function (`x * sigmoid(x)`) and sequence length normalization are computed directly on the accumulator registers of the first dot product. This keeps the data in the high-speed register file/SRAM, eliminating the latency of separate load/store operations for the activation step.
- **efficient masking**：Causal masking is applied via `tl.where` immediately after the activation calculation but before the second dot product. This ensures that masked-out positions contribute zero to the final accumulation without expensive branching or global memory updates.
- **dtype alignment**：The intermediate activation result is cast to `v.dtype` before the second dot product. This ensures the second matrix multiplication utilizes the hardware's highest throughput data type (e.g., FP16/BF16) for the input operands, maximizing the utilization of the compute units (Cube units).
- **block ptr abstraction**：The kernel utilizes tl.make_block_ptr for inputA, inputB, inputC, and Bias tensors. This abstracts the 2D memory access patterns, allowing the compiler to generate efficient vectorized load/store instructions and handle boundary conditions (padding/masking) automatically compared to manual pointer arithmetic.
- **register rematerialization**：Offsets for the output store operation (offs_v_d, offs_m, out_ptrs) are recalculated after the main compute loop rather than being kept in registers throughout. This significantly reduces register pressure during the heavy accumulation phase, preventing register spills and improving occupancy.
- **loop tiling data reuse**：The kernel implements a loop over start_n (sequence dimension) while keeping the inputA block loaded in registers/smem. This tiling strategy maximizes data reuse of the inputA block, reducing the memory bandwidth pressure for loading inputA repeatedly.
- **compile time specialization**：Flags like CAUSAL, HAS_BIAS, and block sizes are passed as tl.constexpr. This allows the Triton compiler to generate specialized kernel variants for different configurations, removing runtime branching overhead inside the critical path.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/006-_hstu_attn_fwd_compute.py
```

**坏实践（反例）**：评分、激活、加权累加拆成多个 kernel 串联，中间 M×N 评分矩阵落 GM
```python
score = tl.dot(a, b)          # kernel1：算出 M×N 评分并写回 GM
score = act_norm_kernel(score)  # kernel2：从 GM 读回做激活+归一化再写回
out = pv_kernel(score, v)       # kernel3：再从 GM 读回做第二次点积
```
后果：中间评分矩阵在 GM 往返至少两次完整读写，带宽开销成为主导，融合收益（35%）全部丢失。✅ 正解：单 kernel 融合——激活与归一化直接在第一次点积的累加器上内联计算，mask 用 `tl.where` 作用于片上数据，中间矩阵不落 GM；第二次点积前把中间结果 cast 到输入 dtype 以跑满 Cube 吞吐。

**已知实例**：attention §raw (`experiences/attention/exp_hstu_attn_fwd_001.yaml`)
- 声称性能收益：35%
- 备注：面向带 seq_offsets 的变长序列注意力评分场景。block pointer 对高效处理 seq_offsets 引入的非规则访存至关重要。

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](../compute-pipeline/exp_cube1_qk_gemm_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](../compute-pipeline/exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](../compute-pipeline/exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
