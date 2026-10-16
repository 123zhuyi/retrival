---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: chunked recurrence strategy"
description: "ssm 场景下，Memory Access Optimization: chunked recurrence strategy，收益 35%"
tags: [访存效率, matrix-operations, ssm]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: chunked recurrence strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，k≥128、v≥128、BT≥64；失效边界：k/v 维度不与 128x64 tile 对齐时切片收益下降，需重选 tile 形状；状态矩阵 k×v 超出 UB/寄存器预算时须增加 tile 数并重算容量；序列短于一个 BT 时 chunk 化与 checkpoint 开销大于收益）

### 原则
- **chunked recurrence strategy**：The kernel implements a chunked processing strategy for the gated incremental-state recurrence. By processing the sequence in blocks of size BT, it reduces the memory pressure and allows for efficient checkpointing of the hidden state h to global memory, which is crucial for long sequences.
- **state matrix tiling**：The hidden state matrix h (inputB x inputC) is explicitly split into two tiles of size 128x64 (b_h1_bv1 and b_h1_bv2). This tiling strategy is designed to fit the working set into the limited high-speed memory (SRAM/Registers) of the AI Core, maximizing data reuse during the iterative update process.
- **block ptr memory access**：Utilizes tl.make_block_ptr for loading keys (k) and storing outputs (h, v_new). This abstracts complex stride calculations and enables the compiler to generate efficient vectorized memory instructions, optimizing global memory bandwidth usage.
- **mixed precision accumulation**：The accumulation of the hidden state (b_h1_bv1, b_h1_bv2) is performed in FP32 to ensure numerical stability during the recurrence, while the heavy computation (tl.dot) is cast to the input data type (fp16/bf16) to leverage the high-throughput matrix units (Cube units).
- **heuristic compilation**：The use of @triton.heuristics allows the kernel to dynamically handle optional features like gating (g), initial states (h0), and variable length sequences (cu_seqlens) without requiring multiple kernel compilations, improving deployment flexibility.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/160-chunk_gated_delta_rule_fwd_kernel_h_blockdim64.py
```

**坏实践（反例）**：逐 token 递推且每步把完整隐状态写回 GM
```python
for t in range(seq_len):
    h = h * decay + tl.dot(k_t, v_t)   # 状态在片上/GM 间往返
    tl.store(h_ptr + t * K * V, h)     # 每步写回整份 K x V 状态
```
后果：隐状态每步一次 GM 写读，长序列下访存量随 seq_len 线性放大，GM 带宽成为瓶颈。✅ 正解：按 BT 分 chunk，chunk 内状态驻留片上（FP32 累加），仅在 chunk 边界 checkpoint 到 GM；状态矩阵按 128x64 切 tile 以放入 UB/寄存器，收益 35%。

**已知实例**：ssm §raw (`experiences/ssm/exp_chunk_gated_delta_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is highly optimized for scenarios where inputB and inputC dimensions align with the 128x64 tiling strategy. It supports variable-length sequences natively via cu_seqlens.

<!-- okf:related:start -->

# 相关

- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 同属优化知识，主题相关
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — chunked recurrence 策略与 chunked WY 骨架同构
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: chunked scan strategy](../compute-pipeline/exp_chunk_cumsum_001.md) — 主题相关：共享 ssm
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 主题相关：共享 matrix-operations, ssm
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 主题相关：共享 matrix-operations, ssm
- 相关主题: [Compute Optimization: vectorized chunked processing](../compute-pipeline/exp_triton_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 ssm, 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 matrix-operations, 访存效率

<!-- okf:related:end -->
