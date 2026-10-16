---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: redundant access elimination"
description: "misc 场景下，Memory Access Optimization: redundant access elimination，收益 25%"
tags: [访存效率, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: redundant access elimination

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32/int32，N ≥1024 的就地 masked 覆写；失效边界：输入为 view 或只读常量时就地写不可用；N 较小、无法摊薄 launch 开销时收益稀释）

### 原则
- **redundant access elimination**：The current implementation writes the input to global memory and then immediately reads it back to compute the masked overwrite. This is optimized by computing the final value `tl.where(mask, value, input)` directly in registers/SRAM, eliminating one global store and one global load per element.
- **vectorization alignment**：The `BLOCK_SIZE_SUB` should be tuned to align with the hardware's burst read length (e.g., 32 bytes or 128 bytes) to ensure memory transactions are fully utilized and to minimize the number of instructions issued for the loop.
- **loop unrolling**：The loop over `num_sub_blocks` can be unrolled or the block size increased to reduce loop overhead, provided register pressure allows, ensuring the pipeline is kept full with memory operations.
- **sub block vectorization**：The kernel implements a two-level tiling strategy where a main blockSize is divided into smaller BLOCK_SIZE_SUB chunks. This allows the memory loads and stores to be precisely aligned with the hardware's vector width (e.g., 128 bytes or specific cache line sizes), significantly improving memory bandwidth utilization and reducing the number of instructions issued.
- **in place memory update**：The kernel performs the read-modify-write operation directly on the input pointer `inp` without allocating an intermediate output buffer. This reduces the total memory traffic by approximately 50% compared to an out-of-place implementation, which is crucial for bandwidth-bound element-wise operations.
- **branchless execution**：The use of `tl.where` to select between the original value and the fill value ensures that the execution path is divergence-free. This avoids the performance penalties associated with conditional branching within the vector core's SIMD lanes.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/143-masked_fill_kernel_self.py
```

**坏实践（反例）**：先把输入整块写到 GM，再读回来做 masked 覆写
```python
tl.store(out_ptr, x)                          # 第一遍：原样写回 GM
x2 = tl.load(out_ptr + offs)                  # 第二遍：再读回来
tl.store(out_ptr + offs, tl.where(mask, value, x2))
```
后果：每元素多一次 GM store 加一次 GM load，访存量近乎翻倍，带宽受限的逐元素场景直接劣化。✅ 正解：在寄存器/片上内存直接用 `tl.where(mask, value, input)` 算出终值并就地写回（read-modify-write 作用于原指针），相比 out-of-place 实现总访存量约降 50%；BLOCK_SIZE_SUB 对齐硬件突发读长度（32B/128B）。

**已知实例**：misc §raw (`experiences/misc/exp_masked_fill_001.yaml`)
- 声称性能收益：25%
- 备注：kernel 依赖 `runtime.get_tuned_config` 选择最优块大小。就地写要求输入张量 `inp` 不是阻止写入的 view 或常量。

<!-- okf:related:start -->

# 相关

- 相关主题: [用 tl.constexpr 编译期消除运行时分支](../compute-pipeline/constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](../compute-pipeline/exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: padding strategy](../compute-pipeline/exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
