---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: operator fusion"
description: "misc 场景下，Memory Access Optimization: operator fusion，收益 35%"
tags: [访存效率, fused-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: operator fusion

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，bf16/fp32，M≤128、H≥1024；失效边界：M>128 时 M×M 矩阵无法整体驻留寄存器/UB，寄存器内迭代失效；H<1024 时 H 维 tile 流水收益稀释；BH 需按目标硬件片上缓存容量重新调优）

### 原则
- **operator fusion**：Fuses RMSNorm, Sinkhorn normalization (MxM matrix ops), and a large-scale weighted sum reduction into a single kernel. This eliminates global memory write/read overhead for intermediate tensors like normalized vectors, the Sinkhorn matrix, and the pre-mix weights.
- **tiled reduction**：Implements a tiling strategy for the weighted sum loop over the `H` dimension. By loading the residual tensor `res_ptr` in tiles of size `[M, BH]`, the kernel ensures data reuse and optimizes memory bandwidth utilization for large hidden dimensions.
- **in register iteration**：The Sinkhorn iteration operates on an `M x M` matrix stored entirely in local variables (`cm`, `rs`, `cs`). This keeps the iterative normalization process in high-speed registers/SRAM, avoiding expensive global memory accesses during the `SR` loop.
- **vectorized accumulation**：The accumulation loops over `NS` (number of sources) use vectorized loads to efficiently aggregate features from multiple sources without branching or scalar operations.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/076-_k_fuse_tlx.py
```

**坏实践（反例）**：归一化、Sinkhorn 迭代、加权求和拆成三个独立 kernel
```python
y  = norm_kernel(x)             # 归一化向量写 GM
cm = sinkhorn_kernel(y)         # M x M 矩阵写 GM，迭代每轮再读回
o  = wsum_kernel(cm, res)       # 预混权重又一次 GM 读写
```
后果：中间张量（归一化向量、M×M 矩阵、预混权重）反复 GM 写读，多次 kernel 启动与带宽开销叠加。✅ 正解：单 kernel 融合三段计算，M×M 矩阵驻留寄存器做迭代，H 维按 [M, BH] tile 流式加载复用，收益 35%。

**已知实例**：misc §raw (`experiences/misc/exp_fuse_tlx_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is optimized for scenarios where M is small (fitting in SRAM) and H is large. The tiling factor `BH` should be tuned to match the L1/L2 cache size of the specific hardware.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_index_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: hardware unit isolation](../compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 fused-operations
- 相关主题: [Scheduling Optimization: deep operator fusion](../compute-pipeline/exp_fused_transformer_block_001.md) — 主题相关：共享 fused-operations, misc
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_grpo_loss_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](../compute-pipeline/exp_launch_diagonal_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](../compute-pipeline/exp_linear_persistent_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_mhc_post_fusion_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: compile time dispatch](../compute-pipeline/exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: scalar loop vectorization](../compute-pipeline/exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
