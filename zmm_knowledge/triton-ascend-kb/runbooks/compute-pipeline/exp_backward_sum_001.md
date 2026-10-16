---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorization alignment'
description: 'reduction 场景下，Compute Optimization: vectorization alignment，收益 25%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, reduction]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorization alignment

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：headDim 维 stride 不为 1（非连续 layout）时合并访存假设失效，需先做 layout 变换；o_len < 128 的小规模场景调大 BLOCK_SIZE_O 反而造成核间负载不均；BLOCK_SIZE_D 未对齐向量宽度时收益消失）

### 原则
- **vectorization alignment**：Align BLOCK_SIZE_D to the hardware's vector width (e.g., 128 for fp16/bf16 on Ascend 910 series) to ensure the Vector Unit processes data efficiently without remainder handling, maximizing throughput for the element-wise multiplication.
- **memory coalescing**：Ensure the stride for the headDim (stride_od, stride_dod) is 1 (contiguous). This allows the memory controller to issue contiguous load instructions, significantly reducing memory access latency compared to strided loads.
- **block size tuning**：Increase BLOCK_SIZE_O to improve data reuse and hide memory latency. By processing more sequence elements per program ID, the kernel can better saturate the compute units while keeping the intermediate accumulation in registers or high-speed cache (L0C/UB).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/021-backward_sum_o_do.py
```

**坏实践（反例）**：BLOCK_SIZE_D 取 100 这类不对齐向量宽度的值，且按非连续 stride 访问 headDim 维 → 每条向量指令都要做 remainder 处理、load 无法合并，吞吐明显下跌 → ✅ 正解即本条：BLOCK_SIZE_D 对齐硬件向量宽度（fp16/bf16 下 128）、保证最内维连续、调大 BLOCK_SIZE_O 提高数据复用。

**已知实例**：reduction §raw (`experiences/reduction/exp_backward_sum_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel computes the gradient for the scale parameter in normalization-like layers. The current implementation is functionally correct but requires tuning of block sizes and alignment to achieve peak performance on Ascend hardware.

<!-- okf:related:start -->

# 相关

- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](../memory-sram/exp_groupnorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: dynamic workload tiling](../memory-sram/exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: persistent kernel strategy](../memory-sram/exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid stride scheduling](../memory-sram/exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](../memory-sram/exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](../memory-sram/exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](../memory-sram/exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column blocking for UB management](../memory-sram/exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column tiling for wide rows](../memory-sram/exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](../memory-sram/exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: block size tuning](../memory-sram/exp_sum_scale_001.md) — 主题相关：共享 reduction, reduction-operations
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
