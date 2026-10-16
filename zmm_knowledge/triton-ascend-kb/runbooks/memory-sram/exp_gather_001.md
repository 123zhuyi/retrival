---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: vectorized indexing"
description: "gather_scatter 场景下，Memory Access Optimization: vectorized indexing，收益 30%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: vectorized indexing

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，num_indices≥32、x_size≥64；失效边界：x_size 过大时整行特征向量驻留片上 UB/寄存器的策略导致寄存器压力过高甚至 spill，需沿特征维再切块；num_indices 很小（<32）时向量化收益被 kernel 启动开销稀释）

### 原则
- **vectorized indexing**：Replace the scalar loop `for i in range(SUB_BLOCK_SIZE)` with vectorized pointer arithmetic. Calculate the base addresses for all rows in the sub-block simultaneously using the `cur_indices` tensor to enable parallel memory access.
- **batched loading**：Utilize `tl.load` with advanced indexing (broadcasted pointer arithmetic) to fetch the entire `(SUB_BLOCK_SIZE, BLOCK_X)` block in a single operation. This eliminates the need for iterative `tl.insert_slice` calls and reduces instruction count.
- **control flow optimization**：Remove the inner Python-level loop within the JIT kernel. This allows the compiler to generate an efficient vectorized instruction stream and improves pipeline utilization by avoiding serial execution of row gathering.
- **vectorized row load**：The kernel loads the entire feature vector (x_vals) for a specific row into local memory/registers in a single operation before performing the gather. This strategy transforms multiple potential scattered global memory reads into one contiguous read, significantly reducing global memory access latency and bandwidth usage.
- **intrinsic gather utilization**：Leverages the tl.gather intrinsic, which maps directly to efficient hardware indexing instructions. This avoids the overhead of complex manual pointer arithmetic or loop-based masking typically required for irregular memory access patterns.
- **stride generalization**：The kernel is designed to handle arbitrary strides for input, index, and output tensors. This flexibility allows it to operate directly on non-contiguous tensor layouts (such as sliced or transposed views) without requiring a preliminary layout transformation (padding or transposition) kernel.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/124-_gather_high_perf_kernel.py
```

**坏实践（反例）**：JIT kernel 内用标量循环逐行 gather，再逐行 insert_slice 拼块
```python
for i in range(SUB_BLOCK_SIZE):            # Python 级串行循环
    row = tl.load(x_ptr + cur_indices[i] * x_stride + tl.arange(0, BLOCK_X))
    out = tl.insert_slice(out, row, (i, 0), (1, BLOCK_X))
```
后果：行 gather 串行执行，指令数随 SUB_BLOCK_SIZE 线性增长，流水线无法打满，不规则访存延迟全部暴露。✅ 正解：用广播指针算术一次性算出子块内所有行的基址，单条 `tl.load` 取回 (SUB_BLOCK_SIZE, BLOCK_X) 整块；或对不规则模式直接用 `tl.gather` 映射硬件索引指令。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_gather_001.yaml`)
- 声称性能收益：30%
- 备注：该方案假设特征维（x_size）足够小，能高效放入片上 UB/寄存器堆。对极大的特征向量，此方法会导致寄存器压力过高，需沿特征维切块。

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
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
