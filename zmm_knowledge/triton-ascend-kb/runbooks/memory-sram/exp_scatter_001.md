---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: vectorized memory access"
description: "gather_scatter 场景下，Memory Access Optimization: vectorized memory access，收益 30%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: vectorized memory access

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；fp16/bf16/fp32；INDICES_LENGTH≥BLOCK_SIZE、NUM_COLUMNS≥BLOCK_X、BLOCK_X≥32；失效边界：索引非唯一且需要原子累加的 scatter 场景失效（本 kernel 假设无原子加，竞态被忽略）；BLOCK_X<32 或 NUM_COLUMNS<BLOCK_X 时向量宽度不足，收益稀释；INDICES_LENGTH<BLOCK_SIZE 时子块切分开销占比升高）

### 原则
- **vectorized memory access**：The kernel loads data from the source tensor `x` in a 2D block (SUB_BLOCK_SIZE x BLOCK_X) and stores to the destination `out` using vectorized stores of width BLOCK_X. This maximizes memory bus utilization compared to scalar operations.
- **sub block tiling**：By processing indices in chunks of SUB_BLOCK_SIZE, the kernel balances register pressure. It keeps a tile of the source matrix in registers/scratchpad while performing the random write pattern characteristic of scatter operations.
- **boundary masking**：Efficient handling of tensor boundaries using `idx_mask` and `col_mask` for both loads and stores. This eliminates the need for padding or expensive conditional branching for partial blocks at the edges of the tensor.
- **hybrid precision compute**：The scaling operation is performed in float32 (`val.to(tl.float32) * scale.to(tl.float32)`) to preserve numerical accuracy during accumulation/multiplication, before casting back to the output type, ensuring stability without sacrificing memory storage efficiency.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/015-_scatter_kernel.py
```

**坏实践（反例）**：逐元素标量化 scatter
```python
for i in range(INDICES_LENGTH):
    for c in range(NUM_COLUMNS):                          # 逐列标量处理
        val = tl.load(x_ptr + src[i]*NUM_COLUMNS + c)     # 标量 load
        tl.store(out_ptr + idx[i]*NUM_COLUMNS + c, val)   # 标量 store
```
后果：标量访存打不满 GM→UB 总线宽度，边界处还需额外分支；向量化方案实测收益 30%。✅ 正解：源张量按 (SUB_BLOCK_SIZE, BLOCK_X) 2D tile 加载、以 BLOCK_X 宽度向量化写回；边界用 `idx_mask`/`col_mask` 掩码处理，避免 padding 和分支；缩放融合在 fp32 寄存器中完成再 cast 回输出类型。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_scatter_001.yaml`)
- 声称性能收益：30%
- 备注：This kernel assumes that the scatter operation does not require atomic adds (i.e., indices are unique or race conditions are acceptable/ignored). The `SCALE` feature is conditionally compiled via the `SCALE` constexpr.

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
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
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
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
