---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: simd index select"
description: "attention 场景下，Memory Access Optimization: simd index select，收益 45%"
tags: [访存效率, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: simd index select

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，BLOCK_DMODEL≥64、BLOCK_N≥32，KV 经 indptr/indices 间接寻址的变长场景；失效边界：KV 连续紧凑存储时 index_select_simd 无收益，普通块加载更优；KV 序列过短时 split_kv 并行收益不足且引入归约开销；依赖 libdevice 内建指令，非 910B 系列平台不可用）

### 原则
- **simd index select**：Utilizes libdevice.index_select_simd to load k/v 张量. This enables parallel index selection from Global Memory to Unified Buffer (UB) with zero-copy semantics, optimizing for the irregular memory access patterns of 变长/分页式 KV 存储场景。
- **online softmax**：Implements the online softmax algorithm (iterative max and sum updates) to avoid materializing the full attention matrix in SRAM/UB, significantly reducing memory bandwidth consumption.
- **kv split parallelism**：Splits the KV sequence dimension across multiple program IDs (split_kv_id), allowing for parallel processing of long sequences and better utilization of compute units.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/027-_fwd_kernel_stage1.py
```

**坏实践（反例）**：用标量循环逐条解析间接索引再访存
```python
for i in range(BLOCK_N):
    kv_idx = tl.load(indices_ptr + start + i)     # 逐条取索引
    k_i = tl.load(k_ptr + kv_idx * stride_k)      # 逐条发起 GM 读
```
后果：不规则访存被串行化，GM→UB 无法并行发起，带宽利用率低。✅ 正解：libdevice.index_select_simd 一次性并行完成索引选择，zero-copy 载入 UB；长序列再叠加 split_kv 并行，收益 45%。

**已知实例**：attention §raw (`experiences/attention/exp_flashattn_gqa_001.yaml`)
- 声称性能收益：45%
- 备注：Optimized for Ascend NPU (910B series) using libdevice intrinsics. Supports variable sequence lengths via indptr/indices.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](../compute-pipeline/exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations, 访存效率

<!-- okf:related:end -->
