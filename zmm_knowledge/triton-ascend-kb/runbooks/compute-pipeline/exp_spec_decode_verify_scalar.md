---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: scalar loop vectorization'
description: 'misc 场景下，Compute Optimization: scalar loop vectorization，收益 300%'
tags: [AICore 单元利用/Scalar 削减, sampling-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: scalar loop vectorization

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：本条描述的是 baseline 缺陷与优化方向（备注自述标量循环是主要瓶颈）；log 空间累加改变数值路径，需验证与线性空间 min(pi*p/q,1) 的等价性；变长需 mask 处理；vocab_size 大时概率 load 的块化收益明显）

### 原则
- **scalar loop vectorization**：The current implementation uses a Python-style scalar loop `for pos in range(num_draft_tokens)` inside the kernel. This serializes the verification of draft tokens, preventing the NPU from utilizing SIMD parallelism for the sequence dimension. The optimization is to vectorize this loop by processing multiple draft positions simultaneously using Triton's vectorized load and arithmetic operations.
- **memory access coalescing**：Inside the scalar loop, `tl.load` is called for individual tokens and probabilities (`draft_token_ids_ptr`, `target_probs_ptr`, etc.). This results in poor memory access patterns and high latency. The optimization involves loading contiguous blocks of data into SRAM before computation, ensuring memory transactions are coalesced.
- **log space accumulation**：The calculation of `pi` involves multiplying probabilities (`pi = min(pi * target_prob / draft_prob, 1.0)`), which can lead to numerical underflow. An optimization is to perform the accumulation in log-space (summing log-probabilities) and converting back to linear space only for the final comparison with the uniform random number.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/171-rejection_random_sample_block_verify_kernel.py
```

**坏实践（反例）**：即本条主题——kernel 内 `for pos in range(num_draft_tokens)` 标量循环 + 逐 token 标量 load → SIMD 完全闲置、访存不合并（潜在收益 300%）→ ✅ 正解：序列维向量化（多位置同时验证）+ 块式合并 load + log 空间累加防概率连乘下溢。

**已知实例**：misc §raw (`experiences/misc/exp_spec_decode_verify_scalar.yaml`)
- 声称性能收益：300%
- 备注：The provided code represents a baseline implementation. The scalar loop over `num_draft_tokens` is the primary bottleneck. Optimizing this requires vectorizing the sequence dimension and handling variable lengths with masks.

<!-- okf:related:start -->

# 相关

- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](../memory-sram/exp_groupnorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc

<!-- okf:related:end -->
