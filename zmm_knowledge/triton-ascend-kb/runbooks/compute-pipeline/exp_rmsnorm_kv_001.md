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
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：归约维（kv_lora_rank）小时权重常驻寄存器的收益有限；权重广播形状 (BLOCK, KV_LORA_RANK) 受寄存器预算约束；fp32 方差计算是精度前提，不可省略）

### 原则
- **grid stride scheduling**：The kernel implements a grid-stride loop using `range(pid, NUM_BLOCKS, NUM_CORES)`. This allows a single program to process multiple blocks of rows, ensuring that all available NPU cores are utilized efficiently even when the total number of rows is not perfectly divisible by the grid size, or when the grid size is smaller than the core count.
- **weight vector caching**：The normalization weights `W` are loaded into registers once before the row processing loop and broadcasted to the shape `(BLOCK, KV_LORA_RANK)`. This optimization eliminates redundant global memory accesses for the weights across different row blocks, significantly reducing memory bandwidth pressure.
- **mixed precision accumulation**：The kernel casts input data `x` to `float32` before computing the variance (`tl.sum(xf * xf, 1)`) and reciprocal square root. This prevents potential overflow or underflow issues inherent in low-precision (fp16/bf16) reduction operations while maintaining high memory throughput by storing the final result back to the original low-precision dtype.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/045-rms_norm_ckv_kernel.py
```

**坏实践（反例）**：每个行块都重新从 HBM load 归一化权重 → 权重被重复读 NUM_BLOCKS 次 → ✅ 正解即本条：权重循环外一次 load 寄存器广播复用 + fp32 方差 + grid-stride 分摊行块。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_kv_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically designed for normalizing the KV LoRA rank dimension within a larger tensor structure, utilizing strided access patterns.

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
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 reduction-operations

<!-- okf:related:end -->
