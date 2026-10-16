---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: software pipelining"
description: "misc 场景下，Memory Access Optimization: software pipelining，收益 25%"
tags: [访存效率, reduction-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: software pipelining

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，n_cols ≥128、total_rows ≥32 的逐行归约场景；失效边界：n_cols <128 时向量宽度利用不足，收益稀释；BLOCK_SIZE 未对齐缓存行（如 128B）时带宽收益下降；num_stages 受 UB 容量约束（p/q 两路流水缓冲 ≤ UB 预算））

### 原则
- **software pipelining**：The kernel utilizes the `num_stages` parameter within `tl.range` to implement software pipelining. This prefetches subsequent blocks of `p` and `q` into the cache while computing the current block, effectively hiding memory access latency.
- **branchless execution**：The gradient calculation uses `tl.where(p > q, 0.5, -0.5)` instead of conditional control flow. This ensures that all lanes within a core execute the same instructions, preventing execution divergence and maintaining high vector-unit utilization.
- **vectorized memory access**：Data is loaded and stored in contiguous chunks defined by `blockSize`. This ensures contiguous GM→UB memory access aligned with the hardware's vector width requirements, maximizing memory bandwidth throughput.
- **efficient masking**：Dynamic masking is applied for both boundary conditions (partial blocks) and the `ignore_index` logic. This allows the kernel to handle variable row lengths and padding without expensive branching or separate kernel launches.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/107-_tv_distance_kernel.py
```

**坏实践（反例）**：load 与计算串行无预取，且用分支判断梯度符号
```python
for off in range(0, n_cols, BLOCK_SIZE):      # tl.range 未设 num_stages
    pv = tl.load(p_ptr + row*n_cols + off + tl.arange(0, BLOCK_SIZE))
    qv = tl.load(q_ptr + row*n_cols + off + tl.arange(0, BLOCK_SIZE))
    if pv > qv:                               # 分支判断，执行分化
        grad = 0.5
    else:
        grad = -0.5
```
后果：下一 block 的 load 延迟无流水遮挡，计算单元空等；分支造成核内 lane 执行分化，向量利用率下降。✅ 正解：`tl.range(..., num_stages=k)` 开启软件流水，预取后续 p/q block 遮挡访存延迟；`tl.where(p > q, 0.5, -0.5)` 无分支计算梯度；连续块 load/store 对齐硬件向量宽度。

**已知实例**：misc §raw (`experiences/misc/exp_tv_distance_001.yaml`)
- 声称性能收益：25%
- 备注：This implementation is highly efficient for per-row reduction tasks. Ensure `blockSize` is tuned to match the specific NPU's L1 cache line size (e.g., 128 bytes) for optimal performance.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_grpo_loss_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: software pipelining](../compute-pipeline/exp_swiglu_bwd_flat_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_topp_filter_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
