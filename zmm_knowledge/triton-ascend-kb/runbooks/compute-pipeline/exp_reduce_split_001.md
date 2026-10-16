---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized memory access'
description: 'reduction 场景下，Compute Optimization: vectorized memory access，收益 20%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, reduction]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized memory access

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：SPLIT_K=1 时无跨 split 归并需求，本 stage 是多余 pass；假设第一阶段已产出部分累加与统计量（备注自述）；head_size_v < 64 时向量块配置需重调）

### 原则
- **vectorized memory access**：The kernel loads the partial accumulation matrix `acc_k` of shape [SPLIT_K, BLOCK_DV] and the statistics vectors `m_k`, `l_k` using vectorized loads. This coalesces memory accesses and significantly improves bandwidth utilization compared to scalar or loop-based loading.
- **numerically stable aggregation**：Implements the online softmax reduction logic by first computing the global maximum `m_max` across all splits and then scaling the partial accumulations by `fast_expf(m_k - m_max)`. This prevents overflow/underflow that can occur when summing exponentials directly.
- **efficient tensor reduction**：Utilizes `tl.sum` with axis=0 to aggregate the scaled partial results along the split-K dimension. This maps efficiently to the hardware's reduction units, minimizing instruction overhead for the summation step.
- **stride aware addressing**：Calculates memory offsets using explicit stride parameters (`stride_ak`, `stride_abs`, etc.), allowing the kernel to support various data layouts (e.g., contiguous or strided heads) without requiring pre-transposition, thus saving memory bandwidth.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/018-_reduce_split_kernel.py
```

**坏实践（反例）**：跨 split 直接对各部分结果的 exp 求和（不做全局 max 对齐）→ 大数值下上溢/下溢 → ✅ 正解即本条：先算全局 `m_max`，各部分按 `fast_expf(m_k - m_max)` 缩放后 `tl.sum(axis=0)` 归并。

**已知实例**：reduction §raw (`experiences/reduction/exp_reduce_split_001.yaml`)
- 声称性能收益：20%
- 备注：This kernel is the second stage of a Split-K attention-like implementation. It assumes the first stage has produced partial accumulations and statistics.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../memory-sram/exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../memory-sram/exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: sub blocking strategy](../memory-sram/exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
