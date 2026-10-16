---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: block ptr abstraction"
description: "attention 场景下，Memory Access Optimization: block ptr abstraction，收益 150%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: block ptr abstraction

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16/fp8，BLOCK_M/BLOCK_N ≥64、HEAD_DIM ≥64；失效边界：本卡为推理专用路径（无反向逻辑、不落 logsumexp），训练/需要反向的场景不适用；HEAD_DIM <64 时块管理开销稀释收益）

### 原则
- **block ptr abstraction**：Utilizes tl.make_block_ptr to handle 2D block loading and storing. This abstracts complex stride calculations and boundary checks, allowing the compiler to generate optimized vectorized memory instructions that maximize memory bandwidth utilization.
- **grid stride scheduling**：Implements a grid-stride loop (for block_idx in range(pid, NUM_BLOCKS, core_step)) to map program IDs to sequence blocks. This ensures robust load balancing across cores, especially when the total number of sequence blocks is not evenly divisible by the grid size.
- **gqa optimization**：Efficiently handles Grouped Query Attention (GQA) by calculating kv_head_num_offset via integer division. This allows multiple query heads to share the same Key and Value heads without duplicating data in memory, significantly reducing memory footprint for GQA models.
- **kernel fusion**：Fuses the QK^T matrix multiplication, exponential operations, row-wise softmax, and final value aggregation into a single kernel. This eliminates the need to write intermediate attention matrices (S and P) back to global memory (HBM), drastically reducing memory traffic.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/217-_sdpa_infer_kernel.py
```

**坏实践（反例）**：评分矩阵 S 与概率矩阵 P 写回 GM 再读回，分组场景还把共享的 K/V 按 query 头复制展开
```python
tl.store(S_ptr, scores)                 # S/P 中间矩阵落 HBM
p = tl.load(S_ptr); p = softmax(p)      # 再读回做归一化
K_full = repeat_kv(K, num_groups)       # K/V 按 query 头数复制，显存膨胀 num_groups 倍
```
后果：中间矩阵的 HBM 往返是带宽主瓶颈；K/V 复制让显存占用与读流量同步放大。✅ 正解：QK^T→softmax→PV 全程融合在单 kernel 内，S/P 永不落盘；分组用 `kv_head = q_head // group_size` 整除映射直接共享 K/V 头，不做数据复制。

**已知实例**：attention §raw (`experiences/attention/exp_sdpa_infer_001.yaml`)
- 声称性能收益：150%
- 备注：This kernel is specifically designed for inference, omitting backward pass logic and storage of logsumexp (m_i) to registers. It supports FP8 data types for the Value tensor.

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
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
