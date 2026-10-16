---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: dynamic specialization with aligned vector load"
description: "rope 场景下，Memory Access Optimization: dynamic specialization with aligned vector load，收益 25%"
tags: [访存效率, matrix-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: dynamic specialization with aligned vector load

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，seq_len >= TOKEN_BLOCK_SIZE 且 head_dim >= rope_dim 的旋转位置编码场景；失效边界：seq_len < TOKEN_BLOCK_SIZE 时 grid-stride 循环退化、调度收益消失；rope_dim 非对齐时走 fallback 路径，单次向量加载的带宽收益消失；`do_not_specialize` 以放弃 seq_len 常量折叠优化为代价，seq_len 固定的静态场景特化编译反而可能更快）

### 原则
- **dynamic specialization**：Uses `do_not_specialize=["seq_len"]` to prevent kernel recompilation for varying sequence lengths, significantly reducing launch overhead in dynamic batching scenarios.
- **memory alignment handling**：Implements an `ALIGNED` path that loads the full RoPE dimension in a single vectorized load, and a fallback path for unaligned dimensions, ensuring maximum memory bandwidth utilization regardless of tensor layout.
- **grid stride scheduling**：Employs a grid-stride loop (`for block_id in range(pid, total_blocks, grid_size)`) to map a fixed number of program instances to a variable number of sequence blocks, maintaining high hardware occupancy.
- **efficient masking**：Combines sequence validity masks with head and dimension masks to perform safe loads/stores for variable-length sequences and KV caches without expensive branching.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/229-_rope_forward_kernel.py
```

**坏实践（反例）**：让 seq_len 参与 kernel 特化
```python
# 未声明 do_not_specialize，seq_len=128/256/512... 每个新值触发一次重编译
def kernel(..., seq_len, ...):   # seq_len 被当作编译期常量
```
后果：动态 batching 下序列长度频繁变化，kernel 反复重编译，launch 开销飙升。✅ 正解：`do_not_specialize=["seq_len"]` 消除重编译；对齐维度走 ALIGNED 路径单次向量加载整 rope_dim；grid-stride 循环用固定数量 program 覆盖可变序列块（收益 25%）。

**已知实例**：rope §raw (`experiences/rope/exp_rope_forward_001.yaml`)
- 声称性能收益：25%
- 备注：Optimized for inference with variable length sequences and KV cache support. Handles both contiguous and non-contiguous memory layouts for RoPE dimensions.

<!-- okf:related:start -->

# 相关

- 相关主题: [用 tl.constexpr 编译期消除运行时分支](../compute-pipeline/constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 同属优化知识，主题相关
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 matrix-operations, 访存效率

<!-- okf:related:end -->
