---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: split k parallelism'
description: 'attention 场景下，Compute Optimization: split k parallelism，收益 35%'
tags: [AICore 单元利用/Scalar 削减, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: split k parallelism

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：kv_len < 1024 时 split-K 引入的第二阶段归并开销超过并行收益；必须配套第二阶段 reduce kernel 聚合部分结果（acc/m_i/l_i），单独使用不完整；BLOCK_H 多头共载仅对分组/多查询头场景有意义，普通等头数场景退化为 1）

### 原则
- **split k parallelism**：The sequence dimension (K 维) is partitioned into SPLIT_K chunks, allowing multiple thread blocks to compute partial attention scores in parallel. This reduces the per-block memory footprint and increases occupancy for long sequences.
- **online softmax**：Implements the iterative online softmax algorithm (updating m_i and l_i) within the main loop. This avoids materializing the full N x M attention matrix in global memory, significantly reducing memory bandwidth consumption.
- **paged kv cache**：Utilizes Block_offsets to load inputB and inputC tiles, supporting non-contiguous memory layouts. This is crucial for inference frameworks (like vLLM) that use paged attention to manage variable-length sequences and KV cache efficiently.
- **gqa head tiling**：The BLOCK_H parameter enables processing multiple query heads simultaneously against a single KV head. This optimization increases data reuse and arithmetic intensity for Grouped Query Attention (GQA) and Multi-Query Attention (MQA) workloads.
- **logit softcapping**：Directly integrates the tanh-based logit softcapping logic (qk = tanh(qk/cap) * cap) into the kernel loop, eliminating the need for a separate post-processing pass for models requiring this feature.
- **head dim vectorization**：Handles head dimensions that are not multiples of the tile size using BLOCK_DMODEL1, ensuring vectorized memory loads and efficient compute utilization for various head sizes (e.g., 128).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/017-_fwd_grouped_split_kernel.py
```

**坏实践（反例）**：长 kv_len 下不做 split，单 program 串行扫完整个序列 → 核间并行度不足、单块片上压力大，长序列时占用率极低 → ✅ 正解即本条：序列维切 SPLIT_K 份并行算部分结果（online softmax 维护 m_i/l_i），交第二阶段归并。

**已知实例**：attention §raw (`experiences/attention/exp_flashattn_splitk_gqa_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel represents the first stage of a split-K attention implementation. A second reduction kernel is required to aggregate the partial outputs (Acc_out, m_i, l_i) produced by each split_k_id.

<!-- okf:related:start -->

# 相关

- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](../algorithm-numerics/split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: kv tile reuse](../memory-sram/exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: fp8 value casting](../memory-sram/exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: sparse gather optimization](../memory-sram/exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: simd index select](../memory-sram/exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: loop invariant code motion](../memory-sram/exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: paged kv cache](../memory-sram/exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, attention-operations
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
