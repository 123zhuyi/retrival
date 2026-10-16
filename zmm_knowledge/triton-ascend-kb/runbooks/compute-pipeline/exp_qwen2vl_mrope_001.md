---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: fused mrope logic'
description: 'rope 场景下，Compute Optimization: fused mrope logic，收益 25%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: fused mrope logic

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：假设输入为标准 2D 布局（rows, head_dim）（备注自述）；hd < 128 时头分块与向量配置需重调；单分量（非多分量 t/h/w）位置编码场景下 `tl.where` 选择逻辑可简化）

### 原则
- **fused mrope logic**：The kernel fuses the complex logic of Multi-RoPE (temporal, height, width) directly into the computation. Instead of preprocessing or splitting tensors, it calculates pointers for t, h, and w components and uses tl.where to dynamically select the correct cos/sin values based on the dimension index. This eliminates extra memory reads and kernel launch overheads associated with handling multi-modal positional encodings.
- **head blocking reuse**：The implementation processes inputA and inputB heads in blocks (BLOCK_Q, blockK). This strategy allows the cos and sin vectors, which are constant for a given row across all heads, to be loaded once from global memory and reused for all heads within the block. This drastically reduces the memory bandwidth pressure for accessing the relatively small positional embedding tables.
- **software pipelining**：The outer loop utilizes the num_stages parameter to implement software pipelining. By prefetching the next row's data while computing the current row, the kernel effectively hides global memory latency, ensuring the compute units are consistently fed with data.
- **vectorized rotation**：The rotation operation is performed by splitting the head dimension into left and right halves (q_left, q_right). This aligns with the memory layout of interleaved RoPE tensors, enabling the use of wide vectorized load/store instructions which maximize memory bus utilization on NPU hardware.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/099-_triton_qwen2vl_mrope_npu.py
```

**坏实践（反例）**：多分量位置编码先拆张量/预处理再分别处理 → 额外内存读写与 kernel launch → ✅ 正解即本条：t/h/w 指针计算 + `tl.where` 按维度索引选 cos/sin 全部融合进主 kernel，cos/sin 在头循环外一次加载复用。

**已知实例**：rope §raw (`experiences/rope/exp_qwen2vl_mrope_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically optimized for multi-modal 3D positional encoding (temporal/height/width) requirements. It assumes the input tensors are in a standard 2D layout (rows, head_dim) and handles boundary conditions for arbitrary sequence lengths and batch sizes.

<!-- okf:related:start -->

# 相关

- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 主题相关：共享 rope
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: sparse iteration logic](../memory-sram/exp_sparse_attn_backward_dq_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope

<!-- okf:related:end -->
