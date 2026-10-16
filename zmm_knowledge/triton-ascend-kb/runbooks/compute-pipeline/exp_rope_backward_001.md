---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: grid stride loop'
description: 'rope 场景下，Scheduling Optimization: grid stride loop，收益 25%'
tags: [多核并行与负载均衡, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: grid stride loop

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：TOKEN_BLOCK_SIZE 需按 L1 容量调（备注自述）；ALIGNED 编译期分支要求对齐前提真实成立，否则拆半加载路径兜底；HAS_KV_LENS 的缓存长度索引对非自回归场景是多余开销）

### 原则
- **grid stride loop**：Implements a grid-stride loop (`for block_id in range(pid, total_blocks, grid_size)`) to allow a single program to process multiple blocks. This increases hardware utilization and load balancing, particularly when the total number of sequence blocks is large or not evenly divisible by the grid size.
- **varlen memory optimization**：Specialized logic for `IS_VARLEN` loads chunk metadata (`chunk_indices_ptr`) to compute dynamic sequence offsets and masks. This avoids padding to the maximum sequence length, significantly reducing memory bandwidth consumption for ragged batches common in LLM inference.
- **vectorized load strategy**：Uses compile-time branching (`ALIGNED`) to select the optimal memory access pattern. When aligned, it loads the full RoPE dimension in a single transaction for maximum throughput; otherwise, it splits loads into two halves to handle unaligned addresses or non-contiguous memory layouts safely without hardware penalties.
- **kv cache aware indexing**：Supports `HAS_KV_LENS` to correctly index `cos` and `sin` tables based on the actual KV cache length per batch. This ensures correctness for autoregressive decoding scenarios where the KV sequence length differs from the current input sequence length.

### 代码/骨架出处
- `TritonPath`:
```python
_rope_backward_kernel
```

**坏实践（反例）**：变长 batch 先 padding 到最大序列长再统一计算 → padding token 空算、带宽浪费 → ✅ 正解即本条：IS_VARLEN 按 chunk 元数据算动态偏移与 mask，只算有效区段 + grid-stride 多块循环。

**已知实例**：rope §raw (`experiences/rope/exp_rope_backward_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is highly optimized for variable-length sequence processing and KV-cache scenarios in Transformer models. Ensure `TOKEN_BLOCK_SIZE` is tuned based on the specific hardware's L1 cache size for optimal performance.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, rope

<!-- okf:related:end -->
