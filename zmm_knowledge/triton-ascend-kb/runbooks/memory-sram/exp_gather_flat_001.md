---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: large block processing"
description: "gather_scatter 场景下，Memory Access Optimization: large block processing，收益 25%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: large block processing

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，N≥4096，索引为 1D 扁平格式（可带 base_offset）；失效边界：N 远小于 BLOCK_SIZE(4096) 时大 block 退化、尾部 mask 占比升高，ILP 收益消失；BLOCK_SIZE×dtype 字节数超过单核 UB 可用容量时需下调 block 重算预算）

### 原则
- **large block processing**：The kernel utilizes a heuristic to set blockSize to 4096, processing a large chunk of elements per program ID. This significantly increases instruction-level parallelism (ILP) and helps hide the high memory latency typically associated with irregular gather operations.
- **stride based addressing**：The input address is calculated efficiently as `base + cur_index * inp_dim_stride`. This allows the kernel to perform gathers along a specific dimension of a multi-dimensional tensor without requiring the input tensor to be flattened or reshaped beforehand, saving memory bandwidth and pre-processing time.
- **vectorized masking**：The kernel employs a mask for both loading indices and storing results. This ensures safe memory access for the tail end of the tensor where the total size N is not a multiple of blockSize, avoiding out-of-bounds errors without expensive conditional branching inside the loop.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/123-_gather_flat_kernel_fixed.py
```

**坏实践（反例）**：用大 N 扁平 gather 却配小 BLOCK_SIZE（如 256）
```python
BLOCK_SIZE = 256
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
idx  = tl.load(idx_ptr + offs, mask=offs < N)
vals = tl.load(inp_ptr + base + idx * inp_dim_stride, mask=offs < N)
```
后果：每个 program 内指令级并行度不足，不规则 gather 的高访存延迟无法被掩盖，需要远更多的 program 数才能填满流水线。✅ 正解：BLOCK_SIZE 取 4096 的经验值，单 program 内大批量元素并行隐藏延迟；索引加载与结果写出都带 mask，安全处理 N 不整除 block 的尾部，无需循环内条件分支。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_gather_flat_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is optimized for flat gather operations where indices are provided in a 1D flattened format. The use of `base_offset` makes it suitable for batched operations or gathering with a base offset.

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: padding strategy](../compute-pipeline/exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
