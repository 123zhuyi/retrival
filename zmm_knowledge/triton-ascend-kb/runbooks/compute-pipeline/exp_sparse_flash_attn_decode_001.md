---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: sparse data access'
description: 'attention 场景下，Compute Optimization: sparse data access，收益 45%'
tags: [AICore 单元利用/Scalar 削减, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: sparse data access

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：依赖 Ascend 专有 intrinsics（tle/cube1），跨平台不可用（备注自述）；稀疏窗口（topk）不稀疏时 gather 开销无回报；手工 workspace 分块需按 L1/L0 容量预算）

### 原则
- **sparse data access**：Utilizes tle.index_select_simd to directly gather Top-K key/value pairs based on Indices_ptr, bypassing the need to load the full KV cache and significantly reducing memory traffic.
- **fused projection**：Integrates the weight matrix multiplication (W_ptr) for both the inputA (split into nope/rope) and the Output accumulation into the attention loop, reducing kernel launch overhead and global memory footprint.
- **npu cube acceleration**：Employs the cube1 intrinsic for the core score 矩阵乘, ensuring maximum utilization of the Ascend NPU's dedicated matrix multiplication units for high-throughput compute.
- **l1 l0 workspace tiling**：Manually manages workspace buffers (Wksp_Q, Wksp_K, Wksp_score, etc.) to keep intermediate tensors (score, Softmax weights) in high-speed cache memory during the iterative reduction process.
- **vectorized sparse loading**：Uses tle.sub_vec_id to distribute the loading of sparse indices and vectors across the vector processing units, optimizing bandwidth for non-contiguous memory access patterns.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/196-sparse_flash_attention_decode_kernel.py
```

**坏实践（反例）**：decode 时加载完整 KV cache 再计算 → 稀疏场景下绝大部分加载的数据被丢弃，带宽全浪费 → ✅ 正解即本条：`tle.index_select_simd` 按索引直接 gather Top-K 对 + 投影矩阵乘融合进循环 + workspace 手工分块驻留 L1/L0。

**已知实例**：attention §raw (`experiences/attention/exp_sparse_flash_attn_decode_001.yaml`)
- 声称性能收益：45%
- 备注：This kernel relies on Ascend NPU specific intrinsics (tle, cube1). It is optimized for decoding scenarios where the attention window is sparse (Top-K selection).

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 attention
- 实践案例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: kv tile reuse](../memory-sram/exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: fp8 value casting](../memory-sram/exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: sparse gather optimization](../memory-sram/exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: simd index select](../memory-sram/exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: loop invariant code motion](../memory-sram/exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: paged kv cache](../memory-sram/exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: sparse iteration logic](../memory-sram/exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention
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
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, attention-operations
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, attention-operations

<!-- okf:related:end -->
