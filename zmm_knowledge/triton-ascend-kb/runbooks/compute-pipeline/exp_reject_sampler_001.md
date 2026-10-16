---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized load'
description: 'misc 场景下，Compute Optimization: vectorized load，收益 35%'
tags: [AICore 单元利用/Scalar 削减, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized load

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：仅适合 max_spec_len ≤ 16 的小固定长度（编译期常量）场景，spec 长度大时全量并行比较的收益被寄存器占用抵消；vocab_size < 32000 时配置需重调；拒绝即停的语义通过 mask 归约表达，需验证与串行版完全等价）

### 原则
- **vectorized load**：Replace scalar loads inside the loop with vectorized loads for `draft_token_ids` and `draft_probs`. Since `max_spec_len` is a compile-time constant and typically small, loading the entire chunk at once significantly reduces memory access overhead and instruction latency compared to iterative scalar loads.
- **parallel comparison**：Remove the sequential `for` loop and the `if not rejected` branch dependency. Calculate the acceptance condition (`target_prob / draft_prob >= random`) for all positions simultaneously using vectorized arithmetic. This transforms the execution from serial to parallel, fully utilizing the SIMD capabilities of the NPU/GPU cores.
- **mask based reduction**：Instead of maintaining a sequential counter `accept_len` with a break condition, generate a boolean mask of accepted tokens and use `tl.sum(mask)` or a prefix scan to determine the final `accept_len`. This leverages hardware-accelerated reduction units and avoids branch divergence.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/224-_reject_sampler_kernel.py
```

**坏实践（反例）**：串行 `for` 循环 + `if not rejected` 分支逐位置判断（原版实现，备注自述）→ 完全串行 + 分支开销 → ✅ 正解即本条：整个 spec chunk 向量化一次 load，接受条件全位置并行计算，mask 归约（`tl.sum`/前缀扫描）求 accept_len。

**已知实例**：misc §raw (`experiences/misc/exp_reject_sampler_001.yaml`)
- 声称性能收益：35%
- 备注：The original implementation relies on a scalar loop with early exit, which limits parallelism and incurs branch overhead. The optimized version processes all speculative steps in parallel, which is particularly effective for the small, fixed `max_spec_len` typical in speculative decoding.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](../memory-sram/exp_embedding_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: vectorized indexing](../memory-sram/exp_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized metadata loading](../memory-sram/exp_moe_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: buffered writes](../memory-sram/exp_padded_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized memory access](../memory-sram/exp_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: binning strategy](../memory-sram/exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, indexing-operations, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc

<!-- okf:related:end -->
