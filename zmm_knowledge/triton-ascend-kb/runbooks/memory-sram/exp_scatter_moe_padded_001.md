---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: sorted bin grouping with padded alignment"
description: "gather_scatter 场景下，Memory Access Optimization: sorted bin grouping with padded alignment，收益 30%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: sorted bin grouping with padded alignment

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；fp16/bf16；NUM_COLUMNS≥64、INDICES_LENGTH≥1024、BLOCK_SIZE≥128；失效边界：bin_ids 未按分组键排序时无法形成连续段，策略失效；分组数少或 INDICES_LENGTH<1024 时分组收益不明显；padded 对齐粒度与硬件访存事务粒度（32/64B）不匹配时对齐收益消失）

### 原则
- **sorted bin grouping**：The kernel leverages sorted bin_ids to group rows by bin within a SUB_BLOCK_SIZE. By calculating bin_count from the first and last element, it processes contiguous segments of the same bin, reducing pointer recalculation overhead and ensuring contiguous GM→UB access patterns for the source tensor x.
- **vectorized memory access**：Utilizes BLOCK_X tiling to process the feature dimension (NUM_COLUMNS) in vectorized chunks. The load operation cur_x = tl.load(...) fetches a block of (SUB_BLOCK_SIZE, BLOCK_X), maximizing global memory bandwidth utilization compared to scalar or small-vector accesses.
- **padding alignment support**：Explicitly handles padded_bins to calculate source offsets. This ensures that memory accesses to the input tensor x are aligned to hardware-friendly boundaries (e.g., 32 or 64 bytes), which is critical for avoiding bank conflicts and achieving peak memory throughput on Ascend hardware.
- **sub block tiling**：Uses a two-level tiling strategy (blockSize for the grid, SUB_BLOCK_SIZE for the loop). This allows the kernel to load a batch of indices and metadata (bin_ids) once, reusing this information while iterating over the feature columns, thus reducing global memory read pressure for metadata.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/033-_padded_scatter_kernel.py
```

**坏实践（反例）**：不按 bin 分组、也不做 padded 对齐，直接按原始乱序索引 scatter
```python
for i in range(INDICES_LENGTH):                  # 乱序索引，同组行散落各处
    row = idx[i]
    src_off = row * NUM_COLUMNS                  # 未对齐的源偏移
    cur = tl.load(x_ptr + src_off + offs_c)      # 地址发散 + 非对齐
    tl.store(out_ptr + dst_off + offs_c, cur)
```
后果：源张量访问地址发散且不对齐，无法发出全宽突发读，指针重算频繁；排序分组+padded 对齐后实测收益 30%。✅ 正解：利用已按分组键排序的 bin_ids，由首末元素算 bin_count 按连续段处理同组行；用 padded_bins 计算对齐到 32/64B 友好边界的源偏移；元数据一次加载后在列循环内复用。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_scatter_moe_padded_001.yaml`)
- 声称性能收益：30%
- 备注：要求输入 bin_ids 已按分组键排序。分组数较多、且 padding 对齐粒度与硬件访存事务粒度匹配时收益最显著。

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: pid grouping strategy](../compute-pipeline/exp_mm_grouped_reorder_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: memory alignment](../compute-pipeline/exp_moe_wgrad_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](row-width-512b-align.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
