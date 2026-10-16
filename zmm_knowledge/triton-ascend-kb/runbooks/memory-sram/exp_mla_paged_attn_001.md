---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: paged kv cache"
description: "attention 场景下，Memory Access Optimization: paged kv cache，收益 45%"
tags: [访存效率, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: paged kv cache

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，bf16/fp16，BLOCK_N≥64、BLOCK_DNOPE≥64、BLOCK_DROPE≥64，KV cache 已按固定大小块预分配且 block_table 由运行时正确填充；失效边界：KV cache 为连续物理布局（无分页）时块表间接寻址纯属多余开销；BLOCK_DNOPE/BLOCK_DROPE<64 时潜维度切片退化；block_table 未填充或页大小与 kernel 假设不一致时结果错误）

### 原则
- **paged kv cache**：Implements a paged KV cache mechanism using a block_table to map logical pages to physical memory blocks. This allows efficient handling of variable-length sequences and reduces memory fragmentation by allocating KV cache in fixed-size blocks.
- **mla decomposition**：Decomposes the attention mechanism into content-based (nope) and positional-based (rope) components. By compressing the Key and Value vectors into a latent dimension (K_NOPE), this drastically reduces the memory bandwidth and storage requirements for the KV cache.
- **online softmax**：Utilizes an iterative online softmax algorithm within the kernel loop. It maintains running maximums (`history_max`) and normalization sums (`l`) to avoid materializing the full $N \times N$ attention matrix, which is critical for long-context inference and memory efficiency.
- **fused gemm pipeline**：Fuses the QK matrix multiplication, softmax calculation, and PV matrix multiplication into a single kernel. This minimizes global memory access by keeping intermediate attention scores and partial accumulators (`acc`) in high-speed SRAM/registers.

### 代码/骨架出处
- `TritonPath`:
```python
_paged_mla_fwd_kernel
```

**坏实践（反例）**：变长序列推理中按最大长度给每条序列连续分配 KV 空间，并把全量注意力分数矩阵物化到 HBM
```python
kv_buf = alloc(max_seq_len, head_dim)      # 每序列按上界连续分配，碎片+浪费
scores = tl.zeros((BLOCK_M, N), dtype=tl.float32)   # 全量分数矩阵
for n_blk in range(kv_blocks):
    scores = q @ k                          # 物化 N×N 分数到 HBM
probs = softmax(scores)
```
后果：连续大分配导致内存碎片、利用率低；物化全量分数矩阵使 HBM 带宽随上下文长度平方增长。✅ 正解：block_table 将逻辑页映射到固定大小物理块，配合在线 softmax（循环内增量维护 history_max/l）与潜维度压缩（键/值压至 K_NOPE），从不物化全量矩阵，取得 45% 收益。

**已知实例**：attention §raw (`experiences/attention/exp_mla_paged_attn_001.yaml`)
- 声称性能收益：45%
- 备注：该类潜空间压缩注意力中，Value（inputC）张量与 Key 内容（K_NOPE）共享同一 buffer。该 kernel 假设 KV cache 已按块预分配、block_table 由运行时正确填充。

<!-- okf:related:start -->

# 相关

- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 attention
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 attention
- 开发指南: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
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
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations, 访存效率

<!-- okf:related:end -->
