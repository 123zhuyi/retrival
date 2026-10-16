---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: sparse iteration logic"
description: "attention 场景下，Memory Access Optimization: sparse iteration logic，收益 40%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: sparse iteration logic

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，HEAD_DIM ≥64、BLOCK_SIZE_K ≥64、TOPK ≥1 且索引已预计算；失效边界：TOPK 接近总序列长度时稀疏迭代退化为稠密 O(N^2)，收益稀释；无预计算 TopK 索引的场景不适用；BLOCK_SIZE_K 过大超出 UB 容量时需重算 tiling）

### 原则
- **sparse iteration logic**：The kernel iterates only over `real_topk` indices derived from the `t_ptr` (topk_idx). This avoids the O(N^2) complexity of standard attention by skipping computation and memory loads for non-essential inputB/inputC blocks, significantly reducing memory bandwidth requirements.
- **persistent data reuse**：Tensors `do` (output gradient), `lse` (log-sum-exp), and `d` (delta) are loaded once per inputA token outside the inner inputB-block loop. These values are held in registers/SRAM and reused for every TopK inputB-block update, minimizing redundant global memory accesses.
- **block ptr efficiency**：Utilizes `tl.make_block_ptr` with `tl.advance` for loading inputB and inputC blocks. This facilitates efficient 2D block memory accesses, handles boundary conditions automatically via `boundary_check`, and allows for padding optimization which aligns with hardware memory transaction sizes.
- **math instruction selection**：Uses `tl.exp2` (base-2 exponential) for the softmax calculation (`p = tl.exp2(qk - lse)`). This is often more efficient on hardware units than natural exponential (`tl.exp`), providing a slight performance boost for the activation step.
- **loop tiling q**：The `num_q_loop` parameter allows a single program instance to process multiple inputA tokens. This increases the arithmetic intensity of the kernel launch and helps amortize the overhead of loading persistent data structures.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/026-backward_dq.py
```

**坏实践（反例）**：仍按稠密方式遍历全部 inputB 块，逐块发起 load 与计算
```python
for kb in range(num_k_blocks):                        # 遍历全部块，含未被 TopK 选中的
    b_blk = tl.load(inputB_ptr + kb*BLOCK_SIZE_K, ...)   # 无效块也发起 GM load
    c_blk = tl.load(inputC_ptr + kb*BLOCK_SIZE_K, ...)
    acc += compute(a_blk, b_blk, c_blk)
    do_blk = tl.load(do_ptr + ...)                    # 常驻数据在内层循环反复加载
```
后果：未被 TopK 选中的块照样消耗访存带宽与算力，访存量回到 O(N^2)，40% 的收益被抹掉；do/lse/d 的重复加载进一步放大冗余 GM 访问。✅ 正解：仅按 `t_ptr`（topk_idx）给出的 `real_topk` 个索引迭代，跳过非关键块的 load 与计算；do/lse/d 在 inputB 块循环外加载一次，驻留寄存器/SRAM 供全部 TopK 块复用。

**已知实例**：attention §raw (`experiences/attention/exp_sparse_attn_backward_dq_001.yaml`)
- 声称性能收益：40%
- 备注：This kernel is designed for variable-length sequences (ragged tensors) utilizing `cu_seqlens`. It assumes pre-computed TopK indices. Performance gains are most significant when TOPK is much smaller than the total sequence length inputB.

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 attention
- 实践案例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: fused mrope logic](../compute-pipeline/exp_qwen2vl_mrope_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
