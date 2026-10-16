---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: sub blocking strategy"
description: "misc 场景下，Memory Access Optimization: sub blocking strategy，收益 25%"
tags: [访存效率, reduction-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: sub blocking strategy

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，HW≥64，NCHW 布局（C 为通道维）；失效边界：HW<64 时 SUB_BLOCK_HW=64 的子块退化为单块，降寄存器压力的收益消失；非 NCHW 布局下 [BLOCK_N, SUB_BLOCK_HW] tile 不再连续，需重写寻址才能保持 GM→UB 连续访存）

### 原则
- **sub blocking strategy**：The spatial dimension HW is processed in fixed-size sub-blocks (SUB_BLOCK_HW=64) within the main loop. This effectively reduces register pressure by limiting the live variables at any given time, preventing register spilling and allowing the compiler to schedule instructions more efficiently.
- **mixed precision accumulation**：Input tensors (X, dY) and statistics (Mean, Rstd) are explicitly cast to float32 before the accumulation operations. This ensures that the summation over potentially large spatial dimensions maintains numerical precision, avoiding overflow or underflow common in fp16/bf16 reductions.
- **vectorized memory access**：The kernel utilizes 2D masking and pointer arithmetic to load tiles of size [blockN, SUB_BLOCK_HW] in a single operation. This maximizes memory bandwidth utilization by keeping GM→UB accesses contiguous and aligned with the hardware's access granularity.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/121-weight_bias_backward_kernel.py
```

**坏实践（反例）**：不做 sub-block，整行 HW 一次性载入累加
```python
offs = tl.arange(0, HW)                    # HW 很大时活跃变量暴增
x  = tl.load(x_ptr + n * HW + offs, mask=offs < HW)
dy = tl.load(dy_ptr + n * HW + offs, mask=offs < HW)
acc += (x * dy).to(tl.float32)
```
后果：活跃变量随 HW 线性增长，寄存器压力过高触发 spill，编译器调度受阻。✅ 正解：主循环内以 SUB_BLOCK_HW=64 的固定子块处理 HW 维，任意时刻活跃变量受限；输入与统计量（Mean、Rstd）显式转 fp32 再累加，避免 fp16/bf16 大维度求和的溢出/下溢。

**已知实例**：misc §raw (`experiences/misc/exp_gn_wb_backward_001.yaml`)
- 声称性能收益：25%
- 备注：该 kernel 面向分组归一化反向（权重/偏置梯度）场景优化。假设 NCHW 数据布局，C 为通道维。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: deep operator fusion](../compute-pipeline/exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_grpo_loss_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_layernorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_reduce_split_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](../compute-pipeline/exp_rmsnorm_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_rmsnorm_rope_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](../compute-pipeline/exp_sda_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_topp_filter_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
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
