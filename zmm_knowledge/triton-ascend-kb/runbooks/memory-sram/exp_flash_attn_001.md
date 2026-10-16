---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: tiling strategy"
description: "attention 场景下，Memory Access Optimization: tiling strategy，收益 200%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-09-22T11:44:30Z
---

## Memory Access Optimization: tiling strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，BLOCK_M≥128、BLOCK_N≥64、HEAD_DIM≥64；失效边界：fp32 累加器 acc（BLOCK_M×HEAD_DIM×4B）与评分块超出 UB 容量时需减小 tile；序列短于一个 BLOCK_N 时分块与在线 softmax 的控制开销大于收益）

### 原则
- **tiling strategy**：Implements 分块注意力 tiling by partitioning the inputA, inputB, and inputC matrices into smaller blocks (blockM, blockN). This ensures that the intermediate attention scores and the output accumulator fit within the fast SRAM, minimizing expensive HBM memory accesses from O(N^2) to O(N^2 / BlockSize).
- **online softmax**：Utilizes the online softmax algorithm to compute attention probabilities incrementally. Instead of materializing the full N x N attention matrix, it maintains running statistics (max value m_i and sum l_i) to update the output accumulator acc in-place, significantly reducing memory footprint.
- **block ptr utilization**：Leverages tl.make_block_ptr for efficient 2D memory loads and stores. This abstraction handles complex stride calculations automatically and enables the hardware to perform vectorized loads, maximizing memory bandwidth utilization for the inputA, inputB, and inputC tensors.
- **fused masking**：Efficiently integrates causal masking and sliding window (BANDWIDTH) logic directly into the computation loop. By applying vectorized boolean masks to the 评分块 scores before the exponential operation, it avoids expensive branching or separate masking kernels.
- **precision management**：Performs accumulation in float32 (acc, m_i, l_i) to maintain numerical stability during the softmax reduction, while supporting fp16/bf16 inputs and outputs. This prevents overflow/underflow without sacrificing the speed of lower-precision tensor core operations.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/088-_attn_fwd.py
```

**坏实践（反例）**：物化完整 N×N 评分矩阵再统一归一化
```python
scores = tl.matmul(q, tl.trans(k))            # N x N 全量评分
p = tl.exp(scores - tl.max(scores))
out = tl.matmul(p / tl.sum(p, ...), v)        # 中间矩阵整体落 GM
```
后果：评分矩阵占 O(N²) GM 空间，HBM 访存量随序列长度平方增长，长序列直接打满带宽。✅ 正解：按 (BLOCK_M, BLOCK_N) 分块，online softmax 维护 m_i/l_i 并原地更新 acc，HBM 访问降至 O(N²/BlockSize)，收益 200%。

**已知实例**：attention §raw (`experiences/attention/exp_flash_attn_001.yaml`)
- 声称性能收益：200%
- 备注：This kernel supports variable sequence lengths via Start_q and sliding window attention via BANDWIDTH. 并包含稳定生成用的注意力汇（sink）处理逻辑。
- CrossformerAttention（分组窗口 MHA，flash_attention，ascend910b，2026-09-21）：非 2 幂 head_dim 时用 **ceil16(BLOCK_D) 替代 next_pow2**（triton-ascend 支持非 2 幂 tl.arange），消除 dot reduction 维虚增（384/448/640/768/896 减 12~60%），逐 case 大 hd 档多减半，Phase 4 汇总 0.0099→0.0118（+19%，唯一确认正向）。证据：`op_15_CrossformerAttention_20260921_1730_3959/output/opt_iter_3/optimized_perf_result.json.speedup_vs_torch`
- ⚠️ 负向（BLOCK 放大在 huge-shape 多 case + 框架噪声下无独立收益）：① 写法 BLOCK_N 16→32（大 G 档想压 KV 迭代）② 边界：49 case 含超大 shape（head_dim≤896、N≤20449）、framework 侧 profiling 噪声 ±2%、小 G 窗口本已 KV 迭代少 ③ 未证伪变体：BLOCK_M 放大（8→16）同样被同轮框架噪声淹没回退（点17）；ceil16 之外的 tiling 增档均无独立收益。证据：`output/opt_iter_4/log.md`、`output/opt_iter_5/log.md`

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](../compute-pipeline/exp_cube1_qk_gemm_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](../compute-pipeline/exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](../compute-pipeline/exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
