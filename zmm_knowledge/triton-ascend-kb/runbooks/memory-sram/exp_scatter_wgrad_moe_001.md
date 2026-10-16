---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: padded binning for grouped scatter reduction"
description: "gather_scatter 场景下，Memory Access Optimization: padded binning for grouped scatter reduction，收益 30%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: padded binning for grouped scatter reduction

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；fp16/bf16；NUM_COLUMNS≥64、BLOCK_SIZE≥128、SUB_BLOCK_SIZE≥16；失效边界：bin_ids 未按分组键排序时连续段不成立，策略失效；各组行数差异小（分布均匀）时分组重排收益有限；未用 padded_bins 对齐时无法发出全宽突发读；SUB_BLOCK_SIZE<16 时向量归约宽度不足）

### 原则
- **grouped binning strategy**：The kernel leverages pre-computed `bin_ids` and `bins` to group rows belonging to the same bin. This transforms the scatter operation into a series of contiguous memory accesses, significantly reducing memory latency and improving cache utilization for the `x` tensor.
- **padded memory alignment**：By utilizing `padded_bins`, the kernel ensures that the memory offsets for each bin's data are aligned to favorable boundaries. This avoids unaligned memory access penalties and allows the memory controller to issue full-width burst reads, which is critical for high-bandwidth hardware like 910B.
- **vectorized reduction**：The inner computation loop loads `BLOCK_X` elements of `x` and `grads` simultaneously to perform a vectorized dot product (`tl.sum(val * grad)`). This maximizes the utilization of vector units and reduces the instruction count per element compared to scalar operations.
- **pointer alignment hints**：The use of `tl.multiple_of` in pointer arithmetic (`grad_ptr`, `x_ptr`) provides explicit hints to the compiler about address alignment, enabling more efficient code generation for global memory loads.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/034-_padded_scatter_wgrad_kernel.py
```

**坏实践（反例）**：不分组 + 逐元素标量归约
```python
acc = 0.0
for i in range(N):                     # 乱序索引，地址发散
    row = idx[i]
    for c in range(NUM_COLUMNS):       # 逐列标量乘加
        acc += tl.load(x_ptr + row*NUM_COLUMNS + c) * tl.load(grad_ptr + row*NUM_COLUMNS + c)
```
后果：标量访存+逐元素归约，指令数随列数线性膨胀，且未分组时源地址发散；分组+向量化后实测收益 30%。✅ 正解：按 bin_ids/bins 把同组行聚成连续段、用 padded_bins 对齐各组偏移；内层一次加载 BLOCK_X 个 `x` 与 `grads` 元素做 `tl.sum(val * grad)` 向量归约；指针运算加 `tl.multiple_of` 对齐提示，让编译器生成更高效的 GM load。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_scatter_wgrad_moe_001.yaml`)
- 声称性能收益：30%
- 备注：专为分组路由反向场景优化（行被路由到不同分组后做梯度归约）。各组行数差异显著时，分组逻辑对性能至关重要。

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
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
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
