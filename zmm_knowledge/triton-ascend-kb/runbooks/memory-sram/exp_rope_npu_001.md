---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: cos sin reuse"
description: "旋转位置编码场景下，Memory Access Optimization: cos sin reuse，收益 35%"
tags: [访存效率, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: cos sin reuse

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；fp16/bf16；hd≥64、sl≥1、每行头数 n_qh/n_kh≥1；失效边界：每行仅单个头（n_qh=n_kh=1）时行间复用无对象，cos/sin 重复加载本就不存在，收益稀释；hd<64 不在适用域；BLOCK_Q×blockK 切片超出 UB 容量时需重算 tiling）

### 原则
- **cos sin reuse**：The cos and sin values are loaded once per row (sequence position) outside the inner head iteration loops. These values are then reused for all heads within that row, significantly reducing redundant global memory accesses and bandwidth pressure.
- **software pipelining**：The outer loop utilizes the `num_stages` parameter in `tl.range` to enable software pipelining. This overlaps the memory loading operations of subsequent rows with the arithmetic computation of the current row, effectively hiding memory latency.
- **head blocking**：The kernel processes query and key heads in fixed-size blocks (`BLOCK_Q` and `blockK`). This tiling strategy ensures that the intermediate data and pointer calculations fit within the NPU's Unified Buffer (UB) or registers, preventing overflow and maximizing on-chip memory utilization.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/104-_triton_rope_npu.py
```

**坏实践（反例）**：在内层头迭代循环里重复加载 cos/sin
```python
for h in range(n_heads):
    cos_b = tl.load(cos_ptr + row_off, ...)   # 每个头重复加载同一行 cos
    sin_b = tl.load(sin_ptr + row_off, ...)   # 每个头重复加载同一行 sin
    # ... 用 cos_b/sin_b 计算第 h 个头
```
后果：同一行（序列位置）的 cos/sin 被重复 GM load n_heads 次，带宽压力随头数线性放大；改为按行加载复用后收益 35%。✅ 正解：在内层头循环之外按行加载一次 cos/sin，行内所有头复用；外层行循环用 `tl.range(num_stages=...)` 软件流水，把后续行的加载与当前行计算重叠以隐藏访存延迟。

**已知实例**：rope §raw (`experiences/rope/exp_rope_npu_001.yaml`)
- 声称性能收益：35%
- 备注：This implementation supports both forward and backward passes (controlled by `BACKWARD_PASS`). It also handles broadcasting of cos/sin when `cos_bs` is 1.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gating_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_partial_rope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_rope_backward_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_rope_triton_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](../compute-pipeline/exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: activation recomputation](../compute-pipeline/exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: software pipelining](../compute-pipeline/exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [循环不变量按复用组外提（cos/sin 类小表每组一次，实测单点 +100%）](../compute-pipeline/loop-invariant-group-hoisting.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations, 访存效率

<!-- okf:related:end -->
