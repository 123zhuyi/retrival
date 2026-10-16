---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: grid stride scheduling'
description: 'normalization 场景下，Scheduling Optimization: grid stride scheduling，收益 25%'
tags: [多核并行与负载均衡, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: grid stride scheduling

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：两遍结构以重读输入换寄存器压力，行宽（n_cols）能单块驻留时一遍即可；FP8/BF16 低精度输入必须配 fp32 上溢保护累加；n_rows < 128 时 grid-stride 无摊薄收益）

### 原则
- **grid stride scheduling**：The kernel utilizes a grid-stride loop (`for row_task_id in range(pid, num_row_tasks, grid_size)`) to process rows. This design allows a single kernel launch to handle matrices where the number of rows exceeds the grid size, ensuring high utilization of compute cores and minimizing kernel launch overhead.
- **two pass algorithm**：The computation is decoupled into two distinct phases a reduction phase to calculate the variance and RSTD, and an element-wise phase for normalization and scaling. This approach significantly reduces register pressure compared to a fused single-pass approach, as it avoids keeping the entire input row in registers during the reduction.
- **adaptive precision accumulation**：The kernel implements flexible casting modes (`_CASTING_MODE_GEMMA`, `_CASTING_MODE_LLAMA`) to upcast inputs to FP32 during accumulation. This is critical for maintaining numerical stability and preventing overflow/underflow when operating on lower precision data types like FP8 or BF16.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/232-_rmsnorm_fwd_kernel.py
```

**坏实践（反例）**：归一化把整行输入驻留寄存器做融合单 pass → 行宽大时寄存器 spill → ✅ 正解即本条：两遍解耦（先归约算 RSTD，再逐元素归一化+缩放）+ grid-stride 适应任意行数。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_fwd_001.yaml`)
- 声称性能收益：25%
- 备注：The grid-stride loop makes this kernel particularly robust for variable batch sizes (n_rows) without requiring re-compilation or host-side grid adjustments.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../memory-sram/exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../memory-sram/exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: sub blocking strategy](../memory-sram/exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 normalization, reduction-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 reduction-operations

<!-- okf:related:end -->
