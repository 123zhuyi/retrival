---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: high precision accumulation'
description: 'misc 场景下，Compute Optimization: high precision accumulation，收益 25%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: high precision accumulation

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：fp32 累加 + 禁 TF32 用吞吐换精度（备注自述：递推状态精度优先的场景才值得）；BT < 16 时 tl.dot 矩阵单元利用率低；无变长需求时 cu_seqlens 路径可裁）

### 原则
- **high precision accumulation**：The kernel enforces float32 accumulation and disables TF32 (allow_tf32=False) during the dot product operations. This is crucial for state recomputation in attention mechanisms to prevent numerical drift, ensuring stability at the cost of raw compute speed.
- **fused gating gemm**：Element-wise scaling operations (multiplying by beta and g) are fused directly into the GEMM pipeline. By computing b_vb and b_kb in registers/SRAM immediately before the tl.dot call, the kernel avoids storing intermediate scaled tensors to global memory, significantly reducing memory bandwidth usage.
- **intra block head iteration**：The kernel loops over the head dimension (H) inside the kernel body rather than launching a separate grid for each head. This reduces kernel launch overhead and can improve cache locality for the attention matrix A if it is shared or accessed sequentially across heads.
- **varlen efficiency**：Implements a specialized path for variable-length sequences using cu_seqlens. This allows the kernel to skip processing padding tokens entirely, maximizing hardware utilization for packed batches common in LLM training/inference.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/156-recompute_w_u_fwd_kernel.py
```

**坏实践（反例）**：缩放后的中间张量（beta/g 加权结果）先写回 HBM 再做 GEMM → 中间张量额外往返 → ✅ 正解即本条：寄存器内算好 `b_vb`/`b_kb` 立即喂 `tl.dot`，头维核内迭代 + cu_seqlens 跳过 padding。

**已知实例**：misc §raw (`experiences/misc/exp_recompute_w_u_001.yaml`)
- 声称性能收益：25%
- 备注：The use of float32 accumulation ensures numerical stability for re-computation but may limit peak throughput compared to pure fp16/bf16 pipelines. Best suited for scenarios where accuracy of the recurrent state is paramount.

<!-- okf:related:start -->

# 相关

- 相关主题: [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](../algorithm-numerics/fp32-precision-contract.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](../memory-sram/exp_rmsnorm_backward_tiled_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: precision control](../memory-sram/exp_skip_layernorm_001.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](../portability/exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](../portability/exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations, misc

<!-- okf:related:end -->
