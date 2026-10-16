---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fused lse reduction"
description: "misc 场景下，Memory Access Optimization: fused lse reduction，收益 25%"
tags: [访存效率, reduction-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fused lse reduction

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp32/bf16，reduction 维 inputC≥1024（典型 32k~128k）的访存瓶颈场景；失效边界：inputC 较小时 kernel 非访存瓶颈，循环融合收益可忽略；三路输入布局不连续或 stride 不一致时，同下标并行读无法形成连续突发）

### 原则
- **fused lse reduction**：The current implementation performs three separate reduction loops over the reduction dimension inputC to calculate the LogSumExp for new, ref, and old logits. These loops should be fused into a single loop. This reduces loop control overhead and improves data locality by processing the same inputC indices for all three tensors simultaneously, allowing for better utilization of memory bandwidth.
- **vectorization alignment**：The BLOCK_SIZE_V should be tuned to align with the hardware's memory burst length (e.g., 128 bytes). For fp16/bf16 data, this typically means a block size of 64 elements. This ensures that all GM→UB accesses are contiguous and aligned, minimizing the number of memory transactions.
- **ilp optimization**：Within the fused reduction loop, utilize Instruction Level Parallelism (ILP) by interleaving the load operations of the current vocabulary chunk with the arithmetic operations (exp, max, sum) of the previous chunk. This helps hide the latency of global memory accesses behind compute operations.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/210-grpo_bwd_kernel.py
```

**坏实践（反例）**：对 new/ref/old 三路 logits 各起一个独立 reduction 循环
```python
for v0 in range(0, V, BLOCK):  lse_new = update(tl.load(new_ptr + v0 + offs))
for v0 in range(0, V, BLOCK):  lse_ref = update(tl.load(ref_ptr + v0 + offs))
for v0 in range(0, V, BLOCK):  lse_old = update(tl.load(old_ptr + v0 + offs))
```
后果：三倍 loop 控制开销，三个循环各自扫一遍 inputC 维，数据局部性差、带宽利用低。✅ 正解：融合为单循环，同一 inputC 下标块同时处理三路张量；BLOCK_SIZE_V 对齐 128B 突发长度（fp16/bf16 取 64 元素）；循环内用 ILP 把当前块的 load 与上一块的 exp/max/sum 算术交叠，隐藏 GM 访存延迟。

**已知实例**：misc §raw (`experiences/misc/exp_grpo_bwd_001.yaml`)
- 声称性能收益：25%
- 备注：该优化对大 reduction 维度（如 inputC=32k 或 128k）、kernel 受内存带宽限制的场景尤为关键。融合循环可减少 reduction 阶段所需的总访存次数。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Scheduling Optimization: deep operator fusion](../compute-pipeline/exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_grpo_loss_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_topp_filter_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率

<!-- okf:related:end -->
