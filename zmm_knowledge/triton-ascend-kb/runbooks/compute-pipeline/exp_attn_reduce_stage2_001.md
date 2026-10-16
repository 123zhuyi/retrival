---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: online softmax algorithm'
description: 'attention 场景下，Compute Optimization: online softmax algorithm，收益 30%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: online softmax algorithm

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：MAX_KV_SPLITS=1（序列不分块）时 online softmax 的 rescale 是纯开销，直接单 pass 归约更优；依赖 kv_indptr 变长索引结构，等长 batch 中可简化；BLOCK_DV < 64 时 tile 过小，向量单元利用率不足）

### 原则
- **online softmax algorithm**：Implements the numerically stable online softmax reduction (max, sum, value update) within a loop over KV splits. This avoids materializing the full [SeqLen, SeqLen] attention matrix, significantly reducing memory bandwidth requirements and HBM access.
- **vectorized accumulation**：Accumulates the output values (`acc`) and normalization statistics (`e_sum`, `e_max`) using float32 precision to maintain numerical stability during exponentiation, while loading inputs potentially in lower precision.
- **split loop optimization**：The loop iterates over `MAX_KV_SPLITS` to aggregate partial results. By calculating `kv_len_per_split` and using `tl.minimum` for bounds checking, it efficiently handles variable sequence lengths without padding, ensuring compute units are not wasted on empty iterations.
- **sink token integration**：Efficiently handles the sink token contribution by integrating it into the final normalization step (`e_sum += tl.exp(cur_sink - e_max)`), avoiding extra passes over the data or complex branching within the main reduction loop.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/028-_fwd_kernel_stage2.py
```

**坏实践（反例）**：第二阶段先把各 split 的部分结果乃至完整 score 矩阵物化到 HBM 再做归约 → [SeqLen, SeqLen] 级中间结果撑爆显存带宽 → ✅ 正解即本条：在 split 循环内做 online softmax（max/sum/输出 rescale），只保留 fp32 的 acc 与统计量。

**已知实例**：attention §raw (`experiences/attention/exp_attn_reduce_stage2_001.yaml`)
- 声称性能收益：30%
- 备注：This kernel is designed for the second stage of two-stage attention-like mechanisms, aggregating partial results from block-wise computation. It relies on `kv_indptr` to handle variable sequence lengths within batches.

<!-- okf:related:start -->

# 相关

- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax](../memory-sram/exp_attn_sinks_001.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: online softmax algorithm](../memory-sram/exp_flashattn_fwd_001.md) — 主题相关：共享 attention
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: dynamic workload tiling](../memory-sram/exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: persistent kernel strategy](../memory-sram/exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: grid stride scheduling](../memory-sram/exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](../memory-sram/exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](../memory-sram/exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](../memory-sram/exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column blocking for UB management](../memory-sram/exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: column tiling for wide rows](../memory-sram/exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](../memory-sram/exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 attention, reduction-operations
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: block size tuning](../memory-sram/exp_sum_scale_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
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
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
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
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
