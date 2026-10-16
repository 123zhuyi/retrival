---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: dynamic vectorization tuning"
description: "normalization 场景下，Memory Access Optimization: dynamic vectorization tuning，收益 25%"
tags: [访存效率, normalization-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: dynamic vectorization tuning

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，HW≥32；失效边界：HW 较小时三遍读 X 的 GM 流量开销无法摊销，单遍融合方案更优；BLOCK_SUB_HW_SIZE×dtype 字节数超过 UB 容量时 autotune 的大配置不可达，搜索空间需按硬件重裁剪）

### 原则
- **dynamic vectorization tuning**：The kernel employs an extensive autotuning configuration for `BLOCK_SUB_HW_SIZE`, sweeping values from 32 to 16384. This allows the runtime to select the optimal vectorization width for the HW dimension, balancing instruction overhead against memory bandwidth saturation and cache capacity for specific tensor shapes.
- **coalesced memory access pattern**：The memory access logic is structured to iterate over the HW dimension (`hw_offsets`) as the inner-most stride. By calculating `base_offset` using `channel_offset * HW + hw_offsets`, the kernel ensures that GM→UB loads and stores are contiguous in global memory, maximizing memory bus utilization.
- **high precision accumulation**：To ensure numerical stability during reduction, the kernel casts input data to `tl.float32` before accumulating `X_sum` and `X_var_sum`. This prevents precision loss or overflow that can occur when summing a large number of elements in lower precision formats (fp16/bf16).
- **three pass algorithm**：The computation is split into three distinct passes (Mean calculation, Variance calculation, and Normalization). While this increases global memory read traffic (loading X three times), it significantly reduces register pressure and simplifies the control flow compared to a single-pass fused approach, enhancing stability for autotuning.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/122-group_norm_kernel.py
```

**坏实践（反例）**：所有 shape 共用一个固定向量宽度，且内层沿非连续维迭代
```python
BLOCK_SUB_HW_SIZE = 512                      # 一刀切，不做 autotune
offs = tl.arange(0, BLOCK_SUB_HW_SIZE) * HW  # 内层按 C 维步进，stride=HW
x = tl.load(x_ptr + channel_offset + offs)
```
后果：GM 访问按 stride=HW 跳跃、完全不连续，总线利用率低；固定宽度在不同 shape 下不是过细（指令开销大）就是过粗（超出 UB）。✅ 正解：内层以 `hw_offsets` 为最内连续维（`base_offset = channel_offset * HW + hw_offsets`），`BLOCK_SUB_HW_SIZE` 在 32~16384 间 autotune 按 shape 选宽；三遍算法降低寄存器压力、简化控制流，累加用 fp32 保精度。

**已知实例**：normalization §raw (`experiences/normalization/exp_groupnorm_001.yaml`)
- 声称性能收益：25%
- 备注：This implementation is optimized for stability and ease of tuning via the three-pass method. It is best suited for scenarios where HW dimensions are large enough to amortize the memory access overhead.

<!-- okf:related:start -->

# 相关

- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: npu simd vectorization](../compute-pipeline/exp_layernorm_npu_simd_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Compute Optimization: scalar loop vectorization](../compute-pipeline/exp_spec_decode_verify_scalar.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: interleaved vectorization](../compute-pipeline/exp_swiglu_oai_001.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
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
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
