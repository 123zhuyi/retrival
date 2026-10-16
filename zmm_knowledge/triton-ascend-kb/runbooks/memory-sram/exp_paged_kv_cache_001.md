---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: head vectorization"
description: "attention 场景下，Memory Access Optimization: head vectorization，收益 40%"
tags: [访存效率, indexing-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: head vectorization

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16，num_kv_heads≥1、head_dim≥64、block_size≥16；失效边界：num_kv_heads 很小（如 1–2）时头维向量化宽度不足，收益稀释；head_dim<64 时连续访存凑不满一个对齐事务；head_dim 非 32B/64B 对齐倍数时 mask 分支开销上升）

### 原则
- **head vectorization**：The original implementation iterates over `num_kv_heads` sequentially using a Python loop. This optimization vectorizes the head dimension by incorporating it into the tensor load/store operations (e.g., expanding pointer arithmetic to handle a range of heads simultaneously). This removes the loop overhead and allows the hardware to process multiple heads in parallel, significantly increasing throughput for models with many heads.
- **memory coalescing**：Optimizes the stride and pointer calculations to ensure that memory accesses along the `head_dim` are contiguous and aligned to the hardware's transaction granularity (e.g., 32 or 64 bytes). This reduces the number of GM transactions required and maximizes bus utilization, which is critical for high-bandwidth memory-bound kernels like KV cache writing.
- **chunk size alignment**：Adjusts the `CHUNK_SIZE` and the internal processing loop to ensure that the amount of data processed per iteration aligns with the hardware's vector width. This prevents partial register writes and ensures that the mask logic does not introduce significant branching overhead.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/238-_store_paged_kv_cache_kernel.py
```

**坏实践（反例）**：Python 循环逐头串行处理 KV cache 写入
```python
for h in range(num_kv_heads):
    vals = tl.load(kv_in_ptr + h * head_dim + d_off)    # 一次只处理一个头
    phys = tl.load(block_table_ptr + blk)
    tl.store(cache_ptr + phys * slot_stride + h * head_dim + d_off, vals)
```
后果：头维循环 overhead 大，每次 load/store 宽度仅 head_dim，多头无法并行，总线利用率低，40% 收益流失。✅ 正解：把头维并入张量 load/store 的指针算术（一次覆盖多个头），并保证 head_dim 方向访存连续、对齐 32/64B 事务粒度，CHUNK_SIZE 对齐硬件向量宽度。

**已知实例**：attention §raw (`experiences/attention/exp_paged_kv_cache_001.yaml`)
- 声称性能收益：40%
- 备注：该优化对 KV 头数量多（如 32、64、128）的分页注意力场景尤为有效，把 kernel 从串行逐头处理转变为宽向量化 store。

<!-- okf:related:start -->

# 相关

- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: npu simd vectorization](../compute-pipeline/exp_layernorm_npu_simd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: scalar loop vectorization](../compute-pipeline/exp_spec_decode_verify_scalar.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: interleaved vectorization](../compute-pipeline/exp_swiglu_oai_001.md) — 同属优化知识，主题相关
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](../portability/exp_rope_matmul_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](layout-bthd-to-bhtd.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
