---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: paged kv indirection"
description: "attention 场景下，Memory Access Optimization: paged kv indirection，收益 35%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: paged kv indirection

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp8，HEAD_DIM≥64、BLOCK_SIZE_M≥64、BLOCK_SIZE_N≥64，且 PAGE_SIZE==BLOCK_SIZE_N；失效边界：PAGE_SIZE≠BLOCK_SIZE_N 时块内边界假设被破坏；解码阶段（单 token）套用 prefill 分块结构收益低；KV cache 连续物理布局时分页间接寻址多余）

### 原则
- **paged kv indirection**：Implements efficient paged KV cache access by utilizing block tables (`block_tables_ptr`) to map logical sequence blocks to non-contiguous physical memory blocks. This allows for dynamic memory management and variable-length sequence handling without memory fragmentation.
- **block ptr vectorization**：Leverages `tl.make_block_ptr` for loading inputA, inputB, and inputC tensors. This API abstracts complex pointer arithmetic and enables hardware vectorization units to load data blocks efficiently, handling boundary checks and padding automatically via `boundary_check` and `padding_option`.
- **online softmax reduction**：Utilizes the online softmax strategy by iteratively updating the accumulator (`acc`), normalization factor (`l_i`), and maximum value (`m_i`) within the loop over KV blocks. This avoids materializing the full $N \times N$ attention matrix in high-bandwidth memory (HBM), significantly reducing memory traffic.
- **gqa support**：Includes logic for 分组查询注意力（多查询头共享一个 KV 头）by calculating `kv_head_id` based on `GQA_INTERLEAVE` flag, allowing efficient mapping of multiple query heads to a single KV head.

### 代码/骨架出处
- `TritonPath`:
```python
paged_prefill_kernel.py
```

**坏实践（反例）**：跨页 KV 手工拼装地址、边界检查与 padding 全部自管
```python
for n_blk in range(num_kv_blocks):
    phys = tl.load(block_tables_ptr + n_blk)           # 逐页取物理块号
    base = phys * PAGE_SIZE * HEAD_DIM
    k = tl.load(kv_ptr + base + offs, mask=hand_mask)  # 手写边界 mask，越界 padding 自管
```
后果：跨页边界与越界 padding 全靠手写 mask，易错且编译器无法合并优化加载，35% 收益流失。✅ 正解：用 `tl.make_block_ptr` 加载 inputA/inputB/inputC，`boundary_check` 与 `padding_option` 自动处理边界；前提 PAGE_SIZE==BLOCK_SIZE_N，循环内在线 softmax 增量更新 acc/l_i/m_i。

**已知实例**：attention §raw (`experiences/attention/exp_paged_prefill_001.yaml`)
- 声称性能收益：35%
- 备注：该 kernel 面向预填充阶段（上下文分块处理）优化。要求 `PAGE_SIZE` 等于 `BLOCK_SIZE_N`。内部 `_sdpa_infer_single_block` 函数承担核心注意力算术。

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
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
