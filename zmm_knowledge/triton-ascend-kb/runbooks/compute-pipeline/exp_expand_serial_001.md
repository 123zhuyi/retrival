---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: compilation specialization control'
description: 'misc 场景下，Scheduling Optimization: compilation specialization control，收益 Baseline'
tags: [多核并行与负载均衡, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: compilation specialization control

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：本条是 baseline 级实现——串行循环 + `tl.get_element` 标量抽取 ILP 极低，仅适合 MAX_NUM_TOKENS ≤ 128 的小 batch 功能场景；高性能场景应重构为 scatter/gather 向量化；do_not_specialize 仅适用于不影响控制流的标量）

### 原则
- **compilation specialization control**：The kernel uses `do_not_specialize` for `replace_from` and `replace_to` arguments. This prevents the compiler from generating a new kernel version for every unique scalar value, reducing compilation overhead and binary size.
- **serial expansion loop**：The core logic relies on a serial loop `for i in tl.range(0, blockSize)` to iterate over the batch elements within a block. This simplifies the handling of variable-length expansions but limits the instruction-level parallelism (ILP) and throughput compared to a fully vectorized approach.
- **scalar register extraction**：Inside the loop, `tl.get_element` is used to extract scalar values (`num_tokens`, `start_idx`, `src_val`) from the vector registers loaded earlier. This allows for per-element calculation of output pointers and masks but introduces overhead associated with scalar extraction from vector lanes.
- **masked vector store**：The output is written using a masked store operation where the mask is dynamically generated based on the `num_tokens` for each specific batch element, ensuring memory safety for jagged arrays.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/169-expand_kernel.py
```

**坏实践（反例）**：不对 `replace_from`/`replace_to` 这类纯数值标量加 `do_not_specialize` → 每个新标量值触发一次 JIT 重编译，编译开销与二进制体积持续膨胀 → ✅ 正解即本条：对纯数值标量禁用特化；同时注意本条的串行循环 + 标量抽取本身是已标注的待优化点。

**已知实例**：misc §raw (`experiences/misc/exp_expand_serial_001.yaml`)
- 声称性能收益：Baseline
- 备注：This implementation represents a naive approach to the expand operation. While functionally correct, the serial loop and scalar extraction (`tl.get_element`) inside the loop are significant performance bottlenecks. For high-performance scenarios, refactoring to remove the loop or utilizing a scatter/gather pattern is recommended.

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](../memory-sram/exp_embedding_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: vectorized indexing](../memory-sram/exp_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: large block processing](../memory-sram/exp_gather_flat_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized metadata loading](../memory-sram/exp_moe_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: buffered writes](../memory-sram/exp_padded_gather_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](../memory-sram/exp_rope_forward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized memory access](../memory-sram/exp_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: binning strategy](../memory-sram/exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: precision control](../memory-sram/exp_skip_layernorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations, misc

<!-- okf:related:end -->
