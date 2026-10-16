---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: block ptr utilization"
description: "attention 场景下，Memory Access Optimization: block ptr utilization，收益 30%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: block ptr utilization

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，HEAD_DIM≥64，解码阶段（每请求上下文定长、批内变长）且 KV cache 为分页布局；失效边界：KV cache 连续物理布局时 block_table 间接寻址不适用；HEAD_DIM<64 时块指针加载退化；预填充阶段（长上下文批量计算）不适用该单 token 解码结构）

### 原则
- **block ptr utilization**：Utilizes tl.make_block_ptr to load inputB and inputC tensors from the paged KV cache. This abstracts complex strided memory accesses and enables the compiler to optimize global memory loads, handling non-contiguous physical memory layouts efficiently.
- **flash attention algorithm**：Implements the online softmax strategy by iteratively updating the max (m_i) and sum (l_i) statistics. This avoids materializing the full attention matrix in HBM, significantly reducing memory bandwidth consumption.
- **paged attention mechanism**：Employs a block table (block_tables_ptr) to map logical sequence blocks to physical memory blocks. This allows for efficient memory management of variable-length sequences and dynamic batching in 分页注意力推理场景.
- **gqa support**：Supports 分组查询注意力（多个查询头共享一个 KV 头）by calculating pid_kh based on NUM_SHARE_Q_HEADS, reducing KV cache memory size and read bandwidth requirements.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/255-paged_decode_kernel.py
```

**坏实践（反例）**：跨页 KV 用手写 strided 指针算术逐页拼装地址、边界 mask 全自管
```python
for blk in range(num_kv_blocks):
    phys = tl.load(block_tables_ptr + blk)             # 逐页取物理块号
    base = phys * PAGE_SIZE * HEAD_DIM
    k = tl.load(kv_ptr + base + offs, mask=hand_mask)  # 手写跨页 strided，边界自管
```
后果：非连续物理页上的手工地址计算易错，编译器无法识别访问模式以优化 GM 加载，边界 mask 指令开销大，30% 收益流失。✅ 正解：用 `tl.make_block_ptr` 抽象跨步访问，由编译器优化全局加载并自动处理边界；配合在线 softmax 不物化全量分数矩阵。

**已知实例**：attention §raw (`experiences/attention/exp_paged_attn_decode_001.yaml`)
- 声称性能收益：30%
- 备注：面向解码阶段优化——每请求上下文定长但批内各请求长度不一。要求 KV cache 为分页布局。

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
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
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
