---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: paged kv cache with gqa head tiling"
description: "attention 场景下，Memory Access Optimization: paged kv cache with gqa head tiling，收益 35%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: paged kv cache with gqa head tiling

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，q_heads_per_kv_head≥1、BLOCK_DMODEL≥64、BLOCK_N≥1；失效边界：q_heads_per_kv_head=1（无头分组）时 BLOCK_H 复用无收益；BLOCK_H 过大使多块查询头中间量超出 UB 容量时需减小；KV cache 连续物理布局时分页间接寻址多余）

### 原则
- **paged kv cache**：Utilizes a block_table to map logical sequence pages to non-contiguous physical memory blocks. This decouples sequence length from physical memory layout, enabling efficient memory management for variable-length sequences and reducing fragmentation during 分页注意力推理.
- **online softmax**：Implements the online softmax algorithm by tracking history_max and the normalization factor l incrementally. This avoids materializing the full N x N attention matrix in global memory or high-bandwidth memory, significantly reducing memory bandwidth consumption.
- **gqa head tiling**：Processes multiple query heads (defined by BLOCK_H) that share the same KV head in a single kernel launch. This optimization amortizes the cost of loading inputB and inputC tiles across multiple query heads, increasing data reuse and arithmetic intensity.
- **latency hiding**：Schedules the load of the Value (inputC) tensor immediately after the Key (inputB) load and before the softmax reduction logic. This overlaps memory access latency with the compute-bound operations of the QK dot product and scaling.
- **accumulator precision**：Maintains intermediate states (acc, l, history_max) in float32 to ensure numerical stability during the exponentiation steps of the softmax, while inputs and outputs can remain in lower precision (fp16/bf16).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/176-_paged_gqa_fwd_kernel.py
```

**坏实践（反例）**：共享同一 KV 头的多个查询头逐个单独加载 K/V tile
```python
for h in range(BLOCK_H):                      # 逐查询头串行处理
    k = tl.load(k_ptr + kv_head_off, ...)     # 每头重复加载同一 K tile
    v = tl.load(v_ptr + kv_head_off, ...)     # 每头重复加载同一 V tile
    acc_h = softmax(q_h @ k) @ v
```
后果：同一 KV tile 被加载 BLOCK_H 次，KV 读带宽放大为 q_heads_per_kv_head 倍，35% 收益流失。✅ 正解：单 kernel 内一次处理共享同一 KV 头的 BLOCK_H 个查询头，KV tile 加载成本摊薄到头组；Value（inputC）load 紧随 Key（inputB）load 调度，与 QK 点积计算重叠掩盖访存延迟。

**已知实例**：attention §raw (`experiences/attention/exp_paged_gqa_001.yaml`)
- 声称性能收益：35%
- 备注：该 kernel 专为大 batch、序列长度各异的解码阶段设计。BLOCK_DPE 参数表明 kernel 加载逻辑内嵌旋转位置编码（rope）分量支持。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention
- 实践案例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 attention
- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 attention
- 实践案例: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../../ops/nsa-selected-attention.md) — 主题相关：共享 attention
- 开发指南: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 attention
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](../portability/exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](layout-bthd-to-bhtd.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
