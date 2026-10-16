---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: loop invariant code motion"
description: "attention 场景下，Memory Access Optimization: loop invariant code motion，收益 30%"
tags: [访存效率, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: loop invariant code motion

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，D ≥64、B ≥1、H ≥1，外积累积式 decode kernel；失效边界：内层维度循环每轮迭代输入张量本就变化时无循环不变量可提；BLOCK_SIZE 不是硬件向量化粒度（32B/64B）整数倍时连续访存收益打折扣）

### 原则
- **loop invariant code motion**：The current implementation redundantly loads the input tensors inside the inner 'pid_d' loop. Since these vectors are constant across the dimension blocks for a specific batch and head, they should be hoisted outside the 'pid_d' loop. This drastically reduces global memory access pressure and instruction count.
- **fused multiply accumulate**：The cache update logic 'kv_outer = kv_outer + ratio * kv_cache_old' represents a classic multiply-add operation. Explicitly ensuring this maps to the hardware's FMA (Fused Multiply-Add) pipeline prevents instruction decoding overhead and maximizes compute unit utilization.
- **grid stride loop optimization**：The kernel utilizes a grid-stride loop ('for pid_bh in range(pid_c, B * H, NUM_CORES)'). To optimize this for the NPU, ensure that the number of launched programs (grid size) is a multiple of the number of physical cores (NUM_CORES) to avoid tail effects and load imbalance.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/069-_lightningattn_attn_decode_kernel.py
```

**坏实践（反例）**：把循环不变量的 load 留在内层分块循环里
```python
for pid_d in range(num_d_blocks):
    a = tl.load(a_ptr + off_a)   # 与 pid_d 无关，却每轮重复从 GM 加载
    b = tl.load(b_ptr + off_b)
    c = tl.load(c_ptr + off_c)
    acc += outer_update(a, b, c, pid_d)
```
后果：内层循环每轮重复加载同一批数据，GM 访问压力与指令数成倍放大，成为 kernel 主要瓶颈。✅ 正解：循环不变量的 load 提升到 pid_d 循环外只加载一次，循环内只保留外积累积更新；grid-stride 的 grid 大小设为 NUM_CORES 整数倍，避免尾部效应与负载不均。

**已知实例**：attention §raw (`experiences/attention/exp_lightning_attn_decode_001.yaml`)
- 声称性能收益：30%
- 备注：该实现的主要瓶颈是分块循环内的冗余内存加载，外提是关键。另外需确认 BLOCK_SIZE 是硬件向量化粒度（32B 或 64B）的整数倍，以启用向量化访存指令。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](../compute-pipeline/exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [循环不变量外提（LICM）减少重复地址计算](../compute-pipeline/loop-invariant-code-motion.md) — 同属优化知识，主题相关
- 相关主题: [循环不变量按复用组外提（cos/sin 类小表每组一次，实测单点 +100%）](../compute-pipeline/loop-invariant-group-hoisting.md) — 同属优化知识，主题相关
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations, 访存效率

<!-- okf:related:end -->
