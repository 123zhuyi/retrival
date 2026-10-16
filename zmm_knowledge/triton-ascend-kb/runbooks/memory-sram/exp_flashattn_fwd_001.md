---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: online softmax algorithm"
description: "attention 场景下，Memory Access Optimization: online softmax algorithm，收益 200%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: online softmax algorithm

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，BLOCK_C≥64、GROUP_SIZE≥1；失效边界：该 micro-kernel 假设查询块 block_q 已预载，仅覆盖内层循环；GROUP_SIZE=1（无头共享）时分组寻址无额外收益；block_q 与输出累加器常驻片上，head_dim 过大超出 UB 容量时需减小 BLOCK_C）

### 原则
- **online softmax algorithm**：Implements 分块注意力的在线归一化策略 by iteratively updating the running maximum (block_m) and the sum of exponentials (block_l). This avoids materializing the large $S \times S$ attention matrix in High Bandwidth Memory (HBM), significantly reducing memory traffic.
- **block tiling strategy**：Loads k/v 张量 in blocks defined by BLOCK_C. This keeps the 查询张量 (block_q) resident in fast on-chip memory (SRAM/Registers) while streaming k and v, maximizing data reuse per byte loaded from HBM.
- **mixed precision computation**：Utilizes LOW_TYPE (e.g., fp16/bf16) for the output accumulation (block_o) to leverage high-throughput tensor core operations, while implicitly using higher precision (HIGH_TYPE) for the softmax statistics (block_s, block_m, block_l) to ensure numerical stability during exponential calculations.
- **gqa memory optimization**：The stride calculation incorporates GROUP_SIZE (idx_n // GROUP_SIZE), enabling efficient memory access patterns for 多个查询头共享同一份 k/v 数据块的分组注意力变体场景。

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/256-micro_kernel_fwd.py
```

**坏实践（反例）**：每个 k/v 块算完就把评分写回 HBM 攒全量矩阵
```python
for c in range(0, S, BLOCK_C):
    s = tl.dot(block_q, k_c)
    tl.store(scores_ptr + c, s)          # S x S 中间矩阵落 HBM
# 第二个 kernel 再读回做归一化与加权求和
```
后果：中间评分矩阵 HBM 写读随 S² 增长，长序列带宽打满。✅ 正解：block_q 常驻片上，流式加载 k/v 块，迭代更新 block_m/block_l 与 block_o，全程无中间矩阵落盘，收益 200%。

**已知实例**：attention §raw (`experiences/attention/exp_flashattn_fwd_001.yaml`)
- 声称性能收益：200%
- 备注：This is a micro-kernel designed for the inner loop of a 分块注意力 forward pass. It assumes block_q is pre-loaded and handles the incremental update of output and softmax statistics.

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
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 主题相关：共享 attention, 访存效率
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
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率

<!-- okf:related:end -->
