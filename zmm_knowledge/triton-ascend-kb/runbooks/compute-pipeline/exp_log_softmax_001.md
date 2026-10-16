---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: two pass algorithm'
description: 'softmax 场景下，Compute Optimization: two pass algorithm，收益 30%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, softmax]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: two pass algorithm

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：N 能单块装下时两遍策略的第二遍重读是纯开销（应单 pass online）；`runtime.get_tuned_config` 依赖调优表覆盖目标 shape；归约维 N 较小时分块循环收益有限）

### 原则
- **two pass algorithm**：The kernel implements a two-pass strategy to handle the reduction over dimension N. The first pass iterates over N in tiles to compute the running maximum (m) and the sum of exponentials (z). The second pass iterates again to apply the log_softmax formula using the reduced values. This decouples the reduction phase from the element-wise computation phase, allowing the kernel to handle sequences (N) much larger than the SRAM/Cache capacity.
- **numerical stability**：It employs the "max trick" (Online Softmax algorithm) to prevent numerical overflow/underflow. By tracking the running maximum `m` and adjusting the exponential calculation as `tl.exp(inp - m_new)`, it ensures that values passed to the exponential function remain within a stable range, even before the final global maximum is known.
- **tiled memory access**：The loop over the reduction dimension N is blocked by `blockN`. This ensures that memory loads from global memory are coalesced and occur in large, contiguous chunks, significantly improving memory bandwidth utilization compared to a scalar or strided access pattern.
- **autotuning configuration**：The kernel utilizes `@triton.autotune` with `runtime.get_tuned_config`. This allows the system to select the optimal `blockM` and `blockN` tile sizes at runtime based on the specific shapes of M and N, ensuring the kernel is performant across a wide range of input dimensions without manual tuning.

### 代码/骨架出处
- `TritonPath`:
```python
log_softmax_kernel.py
```

**坏实践（反例）**：对超长的归约维 N 一次性装入计算 `exp` → 超出 SRAM/寄存器容量直接失败或 spill；且不减 max 直接 exp 上大值溢出 → ✅ 正解即本条：blockN 分块 + 第一遍 online max/sum（max trick），第二遍应用公式。

**已知实例**：softmax §raw (`experiences/softmax/exp_log_softmax_001.yaml`)
- 声称性能收益：30%
- 备注：This implementation is optimized for scenarios where the reduction dimension N is large. The use of `pid_k` allows the kernel to batch process across the K dimension, effectively treating the input as a batch of matrices.

<!-- okf:related:start -->

# 相关

- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](../memory-sram/exp_gn_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](../memory-sram/exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](../memory-sram/exp_rmsnorm_grad_dx_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](../memory-sram/exp_rmsnorm_infer_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 softmax
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 softmax
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 同属优化知识，主题相关
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
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations, softmax

<!-- okf:related:end -->
