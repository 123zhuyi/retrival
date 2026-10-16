---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: two pass reduction strategy"
description: "softmax 场景下，Memory Access Optimization: two pass reduction strategy，收益 30%"
tags: [访存效率, activation-operations, softmax]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: two pass reduction strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，M ≥1、N ≥128、K ≥128，归约发生在非最内层维 N；失效边界：N 装入单 tile 时循环结构无意义（ONE_TILE_PER_CTA 直接单载单算）；归约沿最内层维的场景应看 inner 版本的卡）

### 原则
- **two pass reduction strategy**：For large reduction dimensions (N), the kernel employs a two-pass approach. The first pass iterates over tiles of N to accumulate the 'scale' (sum of output * gradient) in float32. The second pass iterates again to compute the final gradient using the accumulated scale. This decouples the reduction dependency from the element-wise computation, preventing register overflow.
- **adaptive tiling heuristic**：The kernel utilizes the 'ONE_TILE_PER_CTA' heuristic to optimize for different tensor sizes. If N fits within a single tile (TILE_N), it bypasses the reduction loop entirely, performing a single load and compute. If N is larger, it automatically switches to the looped tiling mode, ensuring optimal performance across varying sequence lengths.
- **high precision accumulation**：The intermediate 'scale' variable is explicitly defined with 'tl.float32'. This ensures that the summation over the N dimension maintains higher precision than the input type (fp16/bf16), which is crucial for numerical stability in softmax backward propagation.
- **连续访存布局**：Data access is strided using 'offsets_n[:, None] * inputB + offsets_k', ensuring that loads and stores 沿 inputB 维保持 GM→UB 连续. This maximizes memory bandwidth utilization on the NPU.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/128-softmax_backward_kernel_non_inner.py
```

**坏实践（反例）**：大 N 场景仍用单遍结构，把归约依赖和 element-wise 计算耦在同一循环里
```python
for k in range(K):
    tile = tl.load(ptr + offs_n[:, None] * inputB + k)  # 整个 [N] 切片驻留寄存器
    dx = tile * (g - scale)      # scale 未归约完，所有 tile 都得活着
```
后果：N 大时寄存器溢出，编译器 spill 到本地内存，性能崩塌。✅ 正解：第一遍只用 fp32 标量累加 scale（output×gradient 之和），第二遍再重扫算最终梯度；归约依赖与 element-wise 计算解耦，寄存器压力与 N 脱钩。

**已知实例**：softmax §raw (`experiences/softmax/exp_softmax_backward_non_inner_001.yaml`)
- 声称性能收益：30%
- 备注：This implementation is specifically optimized for the "non-inner" dimension case where the reduction happens along N. It effectively handles the boundary conditions where N is not a multiple of TILE_N via masking.

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 softmax
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 softmax
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 softmax
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 softmax
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 softmax
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gating_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 主题相关：共享 softmax
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](../compute-pipeline/exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, softmax
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, softmax, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations, 访存效率

<!-- okf:related:end -->
