---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: sparse gather optimization"
description: "attention 场景下，Memory Access Optimization: sparse gather optimization，收益 40%"
tags: [访存效率, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: sparse gather optimization

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，HEAD_DIM ≥64、BLOCK_SIZE ≥64，且 topk 索引与块计数已预计算；失效边界：稠密场景（无稀疏索引）时 gather 间接寻址是纯开销；索引未预计算或稀疏率极低（趋近稠密）时不适用）

### 原则
- **sparse gather optimization**：The kernel utilizes `topk_q_idx` (tq_ptr) to dynamically gather only the relevant inputA tokens, DO, LSE, and D values for the current inputB block. This selective loading significantly reduces memory bandwidth consumption and computational workload compared to dense attention, specifically targeting sparse or Top-inputB attention patterns.
- **block ptr tiling**：Employs `tl.make_block_ptr` for the inputB, inputC, DK, and DV tensors. This allows for efficient 2D block loading from global memory to SRAM, automatically handling boundary checks and padding for variable sequence lengths, which simplifies the code and improves hardware utilization.
- **kv data reuse**：Optimizes data locality by loading the inputB and inputC blocks into SRAM once per outer loop iteration (per inputB-block). These blocks are then reused across all active inputA blocks in the inner loop, minimizing redundant global memory accesses and maximizing arithmetic intensity.
- **fp32 accumulation strategy**：Performs all intermediate computations—including attention scores (QK), softmax probabilities (p), delta (ds), and gradient accumulation (dk, dv)—in FP32. This ensures numerical stability during the backward pass, with final results cast to the input data type (e.g., fp16/bf16) only before storing to global memory.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/025-backward_dkdv.py
```

**坏实践（反例）**：稀疏场景仍按稠密方式加载全部 inputA token 再掩码
```python
for start_m in range(0, seq_len, BLOCK_M):
    a_tile = tl.load(inputA_ptr + ...)   # 加载所有行，稀疏位置算完再丢弃
```
后果：被掩码丢弃的位置照样消耗 GM 读带宽和计算量，稀疏率越高浪费越大。✅ 正解：用预计算的 `topk_q_idx` 只 gather 当前 inputB 块相关的 inputA token，DO/LSE/D 同步选择性加载。

**已知实例**：attention §raw (`experiences/attention/exp_attn_sparse_bwd_001.yaml`)
- 声称性能收益：40%
- 备注：This kernel is designed for Top-inputB or Local Attention mechanisms where the attention matrix is sparse. It requires pre-computed indices (`topk_q_idx`) and block counts (`cu_topk_q_count`).

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 attention
- 实践案例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](../compute-pipeline/exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
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
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, 访存效率

<!-- okf:related:end -->
