---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: block pointer utilization"
description: "embedding 场景下，Memory Access Optimization: block pointer utilization，收益 25%"
tags: [访存效率, indexing-operations, embedding]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: block pointer utilization

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，int32 索引，c_len>0、BLOCK_SIZE_N ≥64；失效边界：地址完全发散的非连续 gather 场景 block ptr 收益消失；序列定长且无 padding 时 ragged 处理与早退逻辑收益稀释）

### 原则
- **block pointer utilization**：The kernel utilizes `tl.make_block_ptr` for both source and destination pointers. This abstraction allows the compiler to generate optimized memory access instructions, ensuring better vectorization and cache line utilization compared to manual pointer arithmetic.
- **ragged tensor handling**：By employing `boundary_check=(0,)` during load and store operations, the kernel efficiently processes variable-length sequences (ragged tensors) without requiring padding to the maximum sequence length. This minimizes memory bandwidth consumption and avoids unnecessary computation on padding tokens.
- **early exit condition**：The implementation includes early exit checks (if `c_len <= 0` and if `pid_n * BLOCK_SIZE_N >= c_len`). This ensures that programs corresponding to empty sequences or out-of-bounds ranges terminate immediately, freeing up core resources for active programs.
- **right padding offset**：The source pointer calculation `(n_len - c_len)` dynamically adjusts the base address to skip right-padding. This allows the kernel to read contiguous valid data directly, optimizing memory access patterns for right-padded inputs common in variable-length sequence scenarios.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/024-save_topk_idx_kernel.py
```

**坏实践（反例）**：手动指针算术 + 按最大长度 padding 处理变长序列
```python
max_len = tl.max(all_lens)
offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
v = tl.load(src + offs, mask=offs < max_len)   # 空序列 program 不早退，padding 区也搬运
tl.store(dst + offs, v, mask=offs < max_len)
```
后果：变长序列被 padding 到最大长度，GM 带宽浪费在 padding 区；空序列与越界 program 空转占用核资源。✅ 正解：`tl.make_block_ptr` + `boundary_check=(0,)` 只搬有效段；`c_len<=0` 或 `pid_n*BLOCK_SIZE_N>=c_len` 时立即退出；源地址用 `(n_len - c_len)` 偏移跳过右侧 padding，直接连续读有效数据。

**已知实例**：embedding §raw (`experiences/embedding/exp_idx_copy_001.yaml`)
- 声称性能收益：25%
- 备注：专为变长序列与累计计数场景设计，如按计数汇聚 top-k 索引的拷贝。

<!-- okf:related:start -->

# 相关

- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 同属优化知识，主题相关
- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 主题相关：共享 embedding
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: pointer base precompute](../compute-pipeline/exp_causal_conv1d_prefill_004.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 embedding, indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 embedding, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率

<!-- okf:related:end -->
