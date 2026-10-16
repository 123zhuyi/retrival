---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: block ptr utilization"
description: "misc 场景下，Memory Access Optimization: block ptr utilization，收益 25%"
tags: [访存效率, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: block ptr utilization

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，int32/int64，q_len ≥ BLOCK_SIZE_N、topk ≥ BLOCK_SIZE_T 的变长序列索引搬运/填充场景；失效边界：q_len/topk 小于对应 BLOCK 时 early-exit 与块内并行失效；等长序列（无 ragged）场景 boundary_check 收益弱化；BLOCK_SIZE_N/T 未按硬件缓存行调优时收益下降）

### 原则
- **block ptr utilization**：The kernel utilizes `tl.make_block_ptr` to construct 2D block pointers for both source and destination tensors. This abstraction allows the compiler to optimize memory access patterns, handling complex strides automatically and enabling vectorized memory loads/stores which significantly improves bandwidth utilization compared to scalar pointer arithmetic.
- **boundary check handling**：By passing `boundary_check=(0, 1)` to the `load` and `store` functions, the kernel efficiently handles the padding of variable-length sequences (ragged tensors). This feature automatically masks out-of-bounds accesses for the last block in a sequence, ensuring memory safety without the overhead of manual mask calculation inside the loop.
- **early exit strategy**：The implementation includes a guard condition `if BLOCK_SIZE_N * pid_n >= q_len return` at the beginning of the kernel. This ensures that program instances corresponding to indices beyond the actual sequence length are exited immediately, preventing unnecessary memory transactions and computation for padded regions.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/023-pad_topk_idx_kernel.py
```

**坏实践（反例）**：标量指针算术逐元素搬运 + 手工 mask + 无提前退出
```python
for n in range(BLOCK_SIZE_N):
    idx = tl.load(src_ptr + row*stride + n, mask=n < q_len, other=0)  # 标量逐元素 load
    tl.store(dst_ptr + row*topk + n, idx, mask=n < q_len)             # 手工 mask 逐元素 store
# 越界 program 无任何 guard，仍空跑完整流程
```
后果：标量指针运算带宽利用率低；循环内手工 mask 增加指令开销；索引越界的 program 实例仍发起无效访存与计算。✅ 正解：`tl.make_block_ptr` 构造 2D 块指针配合 `boundary_check=(0, 1)`，越界访问自动掩蔽、无需手工 mask；kernel 开头加 `if BLOCK_SIZE_N * pid_n >= q_len: return`，越界 program 立即退出。

**已知实例**：misc §raw (`experiences/misc/exp_topk_pad_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is optimized for scenarios involving variable-length sequences (e.g., batched ragged-sequence processing) where top-k indices need to be gathered and padded. Ensure BLOCK_SIZE_N and BLOCK_SIZE_T are tuned to the specific cache sizes of the Ascend hardware for optimal performance.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: padding strategy](../compute-pipeline/exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
