---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: single-pass tiling via UB residency"
description: "normalization 场景下，Memory Access Optimization: single-pass tiling via UB residency，收益 40%"
tags: [访存效率, normalization-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: single-pass tiling via UB residency

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，N >= 1024 的归一化梯度场景；失效边界：tile 或分块必须能驻留 UB：BLOCK_SIZE × dtype 字节数 × 驻留份数 ≤ UB 预算，N 极大时分块大小需在避免 UB 溢出与最大化数据复用之间权衡；N 较小、两遍与单遍的带宽差距有限时收益稀释）

### 原则
- **Single-Pass Tiling**：The baseline implementation iterates over the input tensor twice (once for reduction statistics `row_sum_stats`, once for gradient calculation `dx`), resulting in 2x global memory bandwidth consumption. The optimization tiles the row data to fit into the on-chip Unified Buffer (UB), allowing the reduction and gradient computation to be completed in a single pass. This reduces global memory read traffic by approximately 50%.
- **Vectorized Memory Access**：Ensures that memory load/store operations utilize the hardware's maximum vector width (e.g., 128-bit or 256-bit) by aligning `blockSize` and pointer offsets. This maximizes the efficiency of the memory bus and reduces the number of instructions issued for data movement.
- **Software Pipelining**：Overlaps memory load instructions with arithmetic computations. By prefetching the next tile while computing the current tile, the kernel effectively hides memory latency and ensures the compute pipeline remains saturated.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/131-rms_norm_grad_dx_kernel.py
```

**坏实践（反例）**：baseline 两遍法——归约统计与梯度计算各自完整读一遍输入
```python
# pass 1: 读 X 算归约统计
stats = reduce(tl.load(X_ptr + offs))
# pass 2: 再读一遍 X 算梯度
dx = f(tl.load(X_ptr + offs), stats)   # X 从 GM 读两次
```
后果：GM 读带宽消耗 2×，访存成为瓶颈。✅ 正解：行数据 tiling 后驻留 UB，归约与梯度计算在单遍内完成，GM 读流量降约 50%（整体收益 40%）；配合对齐 blockSize 的向量化访存与软件流水隐藏延迟。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_grad_dx_001.yaml`)
- 声称性能收益：40%
- 备注：The optimization assumes that the tile size (N) or the chunked block size fits within the available on-chip memory (UB). For extremely large N, chunking must be carefully managed to avoid UB overflow while still maximizing data reuse.

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: npu simd vectorization](../compute-pipeline/exp_layernorm_npu_simd_001.md) — 主题相关：共享 normalization, normalization-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 normalization
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
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, normalization-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
