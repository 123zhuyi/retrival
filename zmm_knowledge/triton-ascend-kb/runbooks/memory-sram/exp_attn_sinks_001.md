---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: online softmax"
description: "attention 场景下，Memory Access Optimization: online softmax，收益 40%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: online softmax

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，D ≥64，分页存储 + 预计算 sink 分数；失效边界：sink 分数未预计算或序列短到无需滑窗时，sink 集成逻辑成为纯开销；非分页的连续 KV 存储场景 block_table 间接寻址无收益）

### 原则
- **online softmax**：Implements the 在线 softmax algorithm within the main loop. By iteratively updating the maximum value (history_max) and the normalization sum (l) using the 're_scale' factor, the kernel avoids materializing the large N x N attention matrix in high-bandwidth memory (HBM), significantly reducing memory access overhead.
- **paged attention**：Utilizes a block table mechanism (block_tables) to map logical page indices to physical memory blocks. This allows the kernel to efficiently handle non-contiguous KV cache storage, which is essential for managing variable-length sequences and dynamic memory allocation in LLM inference without memory fragmentation.
- **attention sinks integration**：Explicitly loads pre-computed attention scores for 'sink' tokens (the first few tokens of a sequence) and incorporates them into the online softmax statistics. The sink scores are initialized into history_max and l, and later re-scaled and added to the final accumulator, ensuring generation stability even when using sliding windows.
- **sliding window optimization**：Calculates the start page (start_page_num) and start KV length (start_kv_len) based on the sliding_window_size. The kernel applies a mask (mask_page) to ignore tokens outside the sliding window, thereby reducing the number of load operations and matrix multiplications for long sequences.
- **head grouping vectorization**：Processes multiple query heads simultaneously (defined by block_group_size) within a single program instance. This batches the computation of QK scores and value accumulation, improving instruction-level parallelism and better utilizing the hardware's vector units compared to processing heads one by one.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/173-attention_sinks_kernel.py
```

**坏实践（反例）**：把完整 N×N 评分矩阵物化到 GM 再做 softmax
```python
scores = tl.load(...)              # 全量 N×N 评分写回 HBM
tl.store(scores_ptr + ..., scores)
m = tl.max(...)                    # 二次读回做 max/sum 归一化
```
后果：长序列下 N×N 矩阵的 HBM 读写成为带宽瓶颈，且显存占用随 N² 增长。✅ 正解：online softmax——循环内用 re_scale 因子增量更新 history_max 与 l，评分矩阵永不落盘。

**已知实例**：attention §raw (`experiences/attention/exp_attn_sinks_001.yaml`)
- 声称性能收益：40%
- 备注：This kernel is specifically optimized for the decoding phase of LLMs. It assumes that sink scores are pre-computed and passed as an argument. The use of tl.dot for matrix multiplication ensures high throughput on supported hardware.

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 attention
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 同属优化知识，主题相关
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 同属优化知识，主题相关
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax](../compute-pipeline/exp_sdpa_infer_ascend_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_online_001.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
