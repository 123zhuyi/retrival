---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: vectorized offset generation"
description: "attention 场景下，Memory Access Optimization: vectorized offset generation，收益 15%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: vectorized offset generation

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/910_93/910_95，mask_size ≥128、BLOCK_M/BLOCK_N ≥64；失效边界：小块（mask_size <128 或 BLOCK <64）时向量化收益被 kernel 启动开销稀释；非因果/非滑窗的任意稀疏掩码需要额外索引张量，本骨架不直接适用）

### 原则
- **vectorized offset generation**：Utilizes broadcasting with tl.arange to compute memory offsets for the entire BLOCK_M x BLOCK_N tile in a single operation, significantly reducing instruction count compared to iterative index calculation.
- **branchless boundary clamping**：Implements the causal offset logic using min/max arithmetic to constrain indices within the valid mask range [-mask_size, mask_size], 避免条件分支导致的核内执行发散.
- **stride agnostic access**：Directly incorporates mask_stride_m and mask_stride_n into the pointer arithmetic, enabling high-performance loading even when the mask tensor is stored in non-contiguous or transposed memory layouts.
- **minimal register footprint**：Explicitly casts the loaded mask to tl.int1 immediately after loading, optimizing register usage and ensuring the data format is ready for efficient bitwise masking operations during the attention score accumulation.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/252-causal_mask_fn.py
```

**坏实践（反例）**：双重循环逐元素计算掩码偏移 + if 分支裁剪边界
```python
for m in range(BLOCK_M):
    for n in range(BLOCK_N):
        off = (m - n) * stride            # 标量索引计算，指令数膨胀
        if off < -mask_size or off > mask_size:  # 分支发散
            ...
```
后果：偏移计算指令数随 tile 面积线性增长，且分支造成执行发散。✅ 正解：`tl.arange` 广播一次生成整块偏移，用 min/max 无分支裁剪到 [-mask_size, mask_size]。

**已知实例**：attention §raw (`experiences/attention/exp_attn_causal_mask_001.yaml`)
- 声称性能收益：15%
- 备注：适用于因果掩码类评分场景，并通过 q_start/kv_start 参数支持滑动窗口变体

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
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
