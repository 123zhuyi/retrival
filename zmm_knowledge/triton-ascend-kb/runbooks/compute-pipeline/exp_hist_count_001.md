---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: register accumulation'
description: 'misc 场景下，Compute Optimization: register accumulation，收益 50%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: register accumulation

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：bin 数少时全局原子加更简单且足够快（备注自述：bin 数大时本法才占优）；BLOCK_SIZE_R×BLOCK_SIZE_N 工作集必须放进寄存器/SRAM；输入索引含 -1 padding 的假设需与上游约定一致）

### 原则
- **register accumulation**：The kernel accumulates partial histogram results in local registers (variable `y`) for a specific range of output bins (`BLOCK_SIZE_R`) before writing to global memory. This strategy significantly reduces the need for expensive atomic operations on global memory, which are common bottlenecks in histogramming algorithms.
- **vectorized comparison**：Leverages Triton's broadcasting capabilities (`x[:, None] == bins[None, :]`) to perform a parallel comparison between a chunk of input indices and a range of bin indices. This maps efficiently to the SIMT architecture, allowing the GPU to compute multiple histogram bin updates in a single cycle.
- **tiling strategy**：Implements a 2D tiling approach where the outer loop iterates over blocks of the output histogram (`BLOCK_SIZE_R`) and the inner loop iterates over chunks of the input sequence (`BLOCK_SIZE_N`). This ensures that the working set fits within the GPU's SRAM and registers, maximizing data reuse and minimizing global memory access latency.
- **masked padding handling**：Efficiently handles variable sequence lengths and padding by combining load masks with a logical check (`valid_mask = x[:, None] >= 0`). This ensures that padding values (represented as -1) are ignored during the counting process without requiring a separate preprocessing kernel to remove padding.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/022-count_kernel.py
```

**坏实践（反例）**：每个输入元素直接对全局直方图做 `atomic_add` → 大量原子操作打满内存子系统，bin 冲突时严重串行化 → ✅ 正解即本条：广播比较（`x[:,None]==bins[None,:]`）+ 寄存器内按 BLOCK_SIZE_R 局部累加，最后一次性写回。

**已知实例**：misc §raw (`experiences/misc/exp_hist_count_001.yaml`)
- 声称性能收益：50%
- 备注：This optimization is particularly effective for sparse attention or MoE (Mixture of Experts) routing scenarios where counting the frequency of specific indices (e.g., selected experts) is required. It outperforms global atomic addition based methods when the number of bins is large.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: sub blocking strategy](../memory-sram/exp_gn_wb_backward_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: fused lse reduction](../memory-sram/exp_grpo_bwd_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: register buffering](../memory-sram/exp_moe_binned_scatter_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: register reuse intermediate](../memory-sram/exp_rmsnorm_add_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: vectorized chunking](../memory-sram/exp_speculative_recovery_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: software pipelining](../memory-sram/exp_tv_distance_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc, reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc, reduction-operations
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc

<!-- okf:related:end -->
