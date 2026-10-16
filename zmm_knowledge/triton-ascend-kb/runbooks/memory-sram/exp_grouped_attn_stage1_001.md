---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: indirect load scalarization"
description: "attention 场景下，Memory Access Optimization: indirect load scalarization，收益 35%"
tags: [访存效率, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: indirect load scalarization

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，seq_len≥128、head_dim≥64、block_n≥64，经索引间接寻址的分页缓存/变长序列场景；失效边界：索引连续（dense）或缓存布局本身连续时，标量化反而慢于向量化连续 load；在 gather 指令高效的硬件上无需此优化）

### 原则
- **indirect load scalarization**：Replaces vectorized gather loads for the inputB tensor with a scalar loop using `tl.get_element` and `tl.insert_slice`. This optimization addresses severe performance degradation caused by non-contiguous memory accesses when indexing into the cache via `kv_indices`, ensuring efficient memory bandwidth usage on hardware with limited gather efficiency.
- **online softmax algorithm**：Implements the iterative online softmax algorithm within the cache loop. By maintaining running statistics (`e_max`, `e_sum`), the kernel avoids materializing the full $N \times N$ score matrix, significantly reducing memory pressure for long sequences.
- **dpe fusion computation**：Fuses the computation of an additional dimension (`BLOCK_DPE`, typically used for positional embeddings or dequantization scales) directly into the main dot product loop. This eliminates the need for separate kernel launches or extra memory reads for this auxiliary data.
- **vectorized v load**：Retains vectorized loading for the inputC tensor using the native stride calculation, contrasting with the scalarized inputB load. This hybrid approach balances the complexity of inputB indexing with the higher throughput requirements of the final accumulation.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/013-grouped_attention_kernel_stage1.py
```

**坏实践（反例）**：对经索引间接寻址的 inputB 仍用向量化 gather load
```python
k = tl.load(b_ptr + kv_indices[:, None] * stride
            + tl.arange(0, BLOCK_N)[None, :])   # 地址随索引发散
```
后果：GM→UB 访问随 `kv_indices` 高度发散、完全不连续，在 gather 效率受限的硬件上带宽急剧劣化。✅ 正解：inputB 改用 `tl.get_element` + `tl.insert_slice` 标量循环逐条装载；inputC 仍保留原生 stride 的向量化 load，混合策略兼顾 inputB 索引复杂度与最终累加的吞吐需求。

**已知实例**：attention §raw (`experiences/attention/exp_grouped_attn_stage1_001.yaml`)
- 声称性能收益：35%
- 备注：该 kernel 面向分组注意力机制的 "Stage 1"（在线 softmax 变体）设计，专门优化分页缓存或变长序列批处理中固有的不规则访存模式。

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
- 相关主题: [Triton-Ascend 移位 load 降级契约（大位移 ~25x 标量化，2D tile ±1 不合并）](../compute-pipeline/triton-ascend-shifted-load-degradation.md) — 同属优化知识，主题相关
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations, 访存效率
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
