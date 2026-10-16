---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: sparse iteration strategy'
description: 'attention 场景下，Compute Optimization: sparse iteration strategy，收益 40%'
tags: [AICore 单元利用/Scalar 削减, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: sparse iteration strategy

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：稀疏度高（topk << 序列长）时收益最大，topk 接近序列长时退化为稠密还多一层索引开销；依赖 t_ptr 索引张量的预先计算；HEAD_DIM/BLOCK_SIZE_D < 64 时矩阵单元利用率不足）

### 原则
- **sparse iteration strategy**：The kernel iterates only over the `real_topk` blocks identified by the `t_ptr` index tensor, rather than the full sequence length. This drastically reduces the number of expensive GEMM operations (score 矩阵乘和加权求和) from $O(N^2)$ to $O(N \cdot \text{TopK})$.
- **block pointer utilization**：Utilizes `tl.make_block_ptr` for inputA, inputB, inputC, and O tensors. This abstracts complex stride calculations, enables efficient 2D block memory access, and handles boundary conditions (padding) automatically via hardware-friendly instructions.
- **online softmax**：Implements the online softmax algorithm within the sparse loop by maintaining running maximums (`m_i`) and log-sum-exp (`lse_i`). This ensures numerical stability without materializing the large attention score matrix, saving significant memory bandwidth.
- **efficient causal masking**：Applies the causal mask directly on the score 矩阵 using `tl.where((pid_q_j >= c + off_k)...)` inside the compute loop. This vectorized approach avoids loading a separate large mask tensor.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/020-forward_kernel.py
```

**坏实践（反例）**：稀疏场景仍遍历完整序列长度计算所有 score → $O(N^2)$ 的 GEMM 量，绝大部分算在会被丢弃的位置上 → ✅ 正解即本条：只按 `t_ptr` 索引迭代 real_topk 个块 + online softmax（m_i/lse_i）+ 循环内向量化 causal mask。

**已知实例**：attention §raw (`experiences/attention/exp_sparse_attn_topk_001.yaml`)
- 声称性能收益：40%
- 备注：Supports variable sequence lengths and Grouped Query Attention (GQA) via NUM_SHARE_Q_HEADS. Best suited for long-context scenarios where sparsity is high.

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
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, attention-operations

<!-- okf:related:end -->
