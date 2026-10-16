---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: batched reduction replacement'
description: 'gather_scatter 场景下，Compute Optimization: batched reduction replacement，收益 35%'
tags: [AICore 单元利用/Scalar 削减, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: batched reduction replacement

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：BLOCK_X 必须是 cache line（32/64B）倍数；indices 若不唯一且要写同地址，需原子或归约处理（备注自述）；fp32 转换最好在 load 后立即做，循环内转换会在关键路径插入转换指令）

### 原则
- **batched reduction replacement**：The current implementation utilizes a Python-level loop `for i in range(0, SUB_BLOCK_SIZE)` to process the loaded tile. This serializes the computation and prevents the compiler from generating efficient vector instructions. This should be replaced with a batched tensor operation, such as `tl.sum(loaded_x * loaded_grads, axis=1)`, to compute the dot products for the entire sub-block in parallel, significantly increasing ALU utilization.
- **memory access alignment**：The kernel loads `x` and `grads` using `BLOCK_X`. It is critical to ensure `BLOCK_X` is a multiple of the hardware's cache line size (e.g., 32 or 64 bytes depending on the specific 910 series configuration) to ensure memory transactions are fully coalesced, reducing global memory latency.
- **data type promotion optimization**：The code performs `.to(tl.float32)` inside the accumulation loop. To reduce instruction overhead, inputs should be cast to float32 immediately after loading, or the hardware's intrinsic support for mixed-precision computation (if available on the target 910 variant) should be leveraged to avoid explicit casting instructions within the critical path.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/016-_scatter_wgrad_kernel.py
```

**坏实践（反例）**：即本条原则 1 自述的当前缺陷——用 Python 级 `for i in range(SUB_BLOCK_SIZE)` 逐元素处理已加载的 tile → 计算串行化，编译器无法生成向量指令，ALU 利用率低 → ✅ 正解：改批量张量操作（`tl.sum(loaded_x * loaded_grads, axis=1)`）并行算整个子块的点积。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_scatter_wgrad_001.yaml`)
- 声称性能收益：35%
- 备注：The primary bottleneck is the scalar loop over `SUB_BLOCK_SIZE`. Converting this to a vectorized batched operation is essential for achieving high performance on the Ascend 910 series. Additionally, ensure that `indices` are unique or handle collisions atomically if the current `tl.store` is intended for a reduction scenario.

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](../memory-sram/exp_embedding_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized indexing](../memory-sram/exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: vectorized metadata loading](../memory-sram/exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: buffered writes](../memory-sram/exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized memory access](../memory-sram/exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: binning strategy](../memory-sram/exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../memory-sram/parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations

<!-- okf:related:end -->
