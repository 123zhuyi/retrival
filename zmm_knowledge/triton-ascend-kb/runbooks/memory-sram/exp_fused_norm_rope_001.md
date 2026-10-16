---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: operator fusion"
description: "normalization 场景下，Memory Access Optimization: operator fusion，收益 25%"
tags: [访存效率, activation-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: operator fusion

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，head_dim ≥64，输入布局 [seq_len, num_heads, head_dim]；失效边界：全部维度都参与旋转（无 NOPE 子空间）时 partial 优化退化为普通融合；输入布局不符时需先 transpose，额外开销抵消融合收益）

### 原则
- **operator fusion**：Fuses the RMSNorm and Rotary Positional Embedding (RoPE) operations into a single kernel. This eliminates the need to write intermediate normalized results to global memory (HBM) and read them back for RoPE, significantly reducing memory bandwidth consumption and latency.
- **partial rope optimization**：Explicitly handles partial RoPE by separating the head dimension into ROPE (rotated) and NOPE (non-rotated) sub-spaces. The kernel only applies the computationally expensive rotation logic (cos/sin multiplication) to the ROPE dimensions, while the NOPE dimensions are simply normalized and scaled, optimizing arithmetic intensity for 采用部分旋转位置编码的架构.
- **precision management**：Utilizes single-precision floating-point (fp32) for the variance calculation (sum of squares and reciprocal square root) to maintain numerical stability during the reduction, while performing the rest of the vector operations in the input precision (e.g., fp16/bf16) to maximize throughput.
- **grid stride scheduling**：Implements a grid-stride loop over the sequence length dimension. This allows the kernel to efficiently process sequence lengths of arbitrary size without requiring a 1:1 mapping between grid blocks and sequence tokens, ensuring high utilization of the compute units (NUM_CORES).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/065-fused_single_norm_and_partial_rope_kernel.py
```

**坏实践（反例）**：归一化与位置编码旋转拆成两个 kernel
```python
x_norm = rmsnorm_kernel(x)          # kernel 1：归一化结果写回 HBM
x_rope = rope_kernel(x_norm, cos, sin)  # kernel 2：再从 HBM 读回做旋转
```
后果：中间归一化结果多一轮 HBM 写+读，带宽翻倍、延迟叠加；且对 NOPE 维度也套用 cos/sin 旋转逻辑，白付算力。✅ 正解：单 kernel 内融合归一化与旋转，head 维拆成 ROPE/NOPE 两个子空间，仅 ROPE 维度执行旋转。

**已知实例**：normalization §raw (`experiences/normalization/exp_fused_norm_rope_001.yaml`)
- 声称性能收益：25%
- 备注：适用于采用部分旋转位置编码（partial RoPE）的场景。假定输入布局为 [seq_len, num_heads, head_dim]

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gating_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_rope_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](../compute-pipeline/exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: activation recomputation](../compute-pipeline/exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: software pipelining](../compute-pipeline/exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](../compute-pipeline/exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](../compute-pipeline/exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations, normalization, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations, 访存效率

<!-- okf:related:end -->
