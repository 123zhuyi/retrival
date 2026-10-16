---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: Paged Memory Access Strategy"
description: "attention 场景下，Memory Access Optimization: Paged Memory Access Strategy，收益 35%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: Paged Memory Access Strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，BLOCK_M≥64、BLOCK_N≥64、head_size≥64；失效边界：head_size 为标准 tile 整数倍时 BLOCK_DMODEL1 余数路径无用；KV cache 连续物理布局时 Block_offsets 分页寻址不适用；无滑窗/softcapping 需求时 tanh 路径引入多余计算）

### 原则
- **Paged Memory Access Strategy**：Utilizes `Block_offsets` to calculate physical addresses for inputB and inputC tensors dynamically. This allows the kernel to handle non-contiguous KV cache blocks (pages) efficiently, which is critical for managing variable-length sequences and memory fragmentation in 分页注意力推理场景.
- **在线 softmax（online softmax）**：Implements the online softmax algorithm by maintaining running statistics `m_i` (maximum logit) and `l_i` (sum of exponentials) in registers/SRAM. The output accumulator `acc` is updated incrementally within the KV loop, avoiding the materialization of the large $N \times N$ attention matrix in HBM and significantly reducing memory bandwidth pressure.
- **Head Dimension Tiling**：Supports arbitrary head dimensions by splitting the load and compute into `BLOCK_DMODEL` and `BLOCK_DMODEL1` (remainder). This ensures the kernel can handle model configurations where the head size is not a multiple of the standard tile size (e.g., 128) without performance degradation or code duplication.
- **Vectorized Masking and Softcapping**：Integrates causal masking, sliding window attention (`window_size`), and logit softcapping (`tanh`) directly into the compute loop using vectorized operations like `tl.where`. This approach minimizes control flow divergence and ensures that masking logic does not introduce significant overhead.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/019-_fwd_kernel.py
```

**坏实践（反例）**：把全量注意力分数矩阵物化到 HBM 再做 softmax
```python
scores = tl.zeros((BLOCK_M, N), dtype=tl.float32)   # 全量分数矩阵
for n_blk in range(kv_blocks):
    k = tl.load(k_ptr + Block_offsets[n_blk], ...)
    scores = q @ k                                   # 分数写入 HBM
probs = softmax(scores)                              # 二次遍历 N×N 矩阵
out = probs @ v
```
后果：N×N 分数矩阵在 HBM 上物化并两次遍历，带宽压力随上下文长度平方增长，35% 收益流失。✅ 正解：在线 softmax——寄存器/SRAM 中维护 `m_i`、`l_i`，KV 循环内增量更新 `acc`，从不物化全量矩阵；head 维非 tile 整数倍时拆 BLOCK_DMODEL+BLOCK_DMODEL1 两段处理。

**已知实例**：attention §raw (`experiences/attention/exp_paged_attn_flash_001.yaml`)
- 声称性能收益：35%
- 备注：通过 `kv_group_num` 参数支持分组查询注意力（多查询头共享 KV 头），并通过 `Q_seqlens`、`KV_seqlens` 处理变长序列。

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
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
