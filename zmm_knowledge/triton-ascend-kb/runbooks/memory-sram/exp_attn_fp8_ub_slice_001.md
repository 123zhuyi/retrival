---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fp8 value casting"
description: "attention 场景下，Memory Access Optimization: fp8 value casting，收益 30%"
tags: [访存效率, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fp8 value casting

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16/fp8，HEAD_DIM ≥256、BLOCK_M/BLOCK_N ≥128；失效边界：HEAD_DIM <256 时无需切片，标准累加器路径更高效（切片反而增加指令开销）；UB 容量不同的硬件需重算切片数）

### 原则
- **fp8 value casting**：The softmax weights p are cast to float8e5 before the PV dot product if fp8_v is enabled. This reduces the data volume for the PV accumulation and leverages FP8 compute acceleration.
- **ub overflow slicing**：For large head dimensions (headDim >= 256), the accumulator is explicitly sliced into 4 segments (blockM // 4) during the update phase. This prevents Unified Buffer (UB) overflow on the NPU by ensuring intermediate results fit within on-chip memory limits.
- **online softmax**：Implements the online softmax algorithm within the loop, updating m_i (max) and l_i (sum) incrementally. This avoids storing the full attention matrix, significantly reducing global memory access.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/011-_attn_fwd_inner.py
```

**坏实践（反例）**：headDim ≥256 时仍用整块累加器更新，不做 UB 切片
```python
acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)  # HEAD_DIM=256+ 时中间结果撑爆 UB
```
后果：中间结果超出 UB 容量，触发编译期资源分配失败或运行期 spill，kernel 不可用。✅ 正解：更新阶段把累加器切成 4 段（blockM // 4）逐段处理，确保中间结果始终留在片上内存限制内。

**已知实例**：attention §raw (`experiences/attention/exp_attn_fp8_ub_slice_001.yaml`)
- 声称性能收益：30%
- 备注：The slicing logic is specifically required for headDim >= 256 to avoid hardware resource limits. For smaller heads, the standard accumulator path is more efficient.

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 同属优化知识，主题相关
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 attention
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](../compute-pipeline/exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
