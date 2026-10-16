---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: compile time branching"
description: "conv 场景下，Memory Access Optimization: compile time branching，收益 25%"
tags: [访存效率, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: compile time branching

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，D ≥32、W ≥4；失效边界：T ≥ BW 时状态累加融合路径不生效（融合仅覆盖 T < BW 的分支）；BD/BW 与 L2 容量不匹配时需重新调参，否则性能明显回落）

### 原则
- **compile time branching**：Utilizes @triton.heuristics to resolve IS_VARLEN and USE_INITIAL_STATE at compile time. This eliminates runtime branch divergence within the kernel, ensuring a streamlined instruction pipeline for specific execution paths.
- **vectorized block io**：Implements 2D vectorized memory access by calculating block pointers for BD (Dimension Block) and BW (Width Block). Using broadcasting (o_d[:, None]), it loads/stores tiles of size [BD, BW] in a single operation, maximizing DRAM bandwidth utilization and reducing the number of load/store instructions.
- **masked boundary handling**：Employs sophisticated masking logic (m_t, m_d, m_w) to handle sequence boundaries and variable lengths (cu_seqlens) safely. This allows the kernel to process the tail end of sequences without out-of-bounds memory access, avoiding the need for input padding or separate edge-case kernels.
- **state accumulation fusion**：Fuses the initial state accumulation logic directly into the kernel when T < BW. By adding the initial state cache (b_cache) to the input block (b_x) before storing, it minimizes global memory read/write operations required for state initialization.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/268-causal_conv1d_states_fwd_kernel.py
```

**坏实践（反例）**：状态初始化拆成独立步骤——先写输入块、再读回加初始状态（或另起 kernel）
```python
tl.store(out_ptrs, b_x, mask=...)              # 先写回
b_x = tl.load(out_ptrs) + tl.load(cache_ptrs)  # 再读回加状态，GM 读写翻倍
```
后果：状态初始化引入成倍的 GM 往返读写。✅ 正解：T < BW 时把初始状态 `b_cache` 在片上直接加进输入块 `b_x` 后一次写出，初始化与主计算融合。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_states_001.yaml`)
- 声称性能收益：25%
- 备注：适用于需要跨批维护与更新卷积状态的序列建模场景。性能高度依赖 BD/BW 与硬件 L2 容量的匹配调参

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](../compute-pipeline/exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](../compute-pipeline/exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](../compute-pipeline/exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](../compute-pipeline/exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: compile time dispatch](../compute-pipeline/exp_radix_convert_001.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
