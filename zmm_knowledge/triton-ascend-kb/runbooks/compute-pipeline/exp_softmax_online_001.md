---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: online softmax algorithm'
description: 'softmax 场景下，Compute Optimization: online softmax algorithm，收益 35%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, softmax]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: online softmax algorithm

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：本条是 loop 内调用的 helper（备注自述），依赖 WS1/WS2 指向高速共享内存；BLOCK_M/BLOCK_N < 64 时块归约效率低；低精度存储中间概率对精度极敏感场景需评估）

### 原则
- **online softmax algorithm**：The kernel implements the online softmax algorithm by iteratively updating the running maximum (m_i) and the running sum of exponentials (l_i). This allows the softmax operation to be fused into the main attention loop without materializing the full $N \times N$ attention matrix in global memory, significantly reducing memory bandwidth pressure.
- **numerical stability**：The calculation `qk - m_ij[:, None]` subtracts the row-wise maximum before the exponential operation. This prevents floating-point overflow or underflow during the `tl.math.exp` calculation, which is critical for the stability of attention mechanisms in deep learning models.
- **mixed precision storage**：The computation is performed in higher precision (implied float32 for accumulation), but the final result `p_cast` is cast to `inputA.type.element_ty` (e.g., fp16/bf16) before storing to `WS2`. This optimizes shared memory usage and store throughput while maintaining sufficient numerical precision for the gradients.
- **vectorized reduction**：The use of `tl.max(qk, 1)` and `tl.sum(p, 1)` leverages hardware vector units to perform the reduction operations efficiently across the block dimension, minimizing the number of instruction cycles required for the reduction step.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/054-vec_a.py
```

**坏实践（反例）**：物化完整 N×N score 矩阵到 HBM 再统一 softmax → 长序列下带宽与容量双爆 → ✅ 正解即本条：循环内 online 更新 m_i/l_i + `qk - m_ij` 减 max 防溢出 + 结果降精度存 WS2 省片上空间。

**已知实例**：softmax §raw (`experiences/softmax/exp_softmax_online_001.yaml`)
- 声称性能收益：35%
- 备注：This function is designed to be called within a loop (e.g., over sequence blocks or heads) in a two-pass attention-style kernel. It relies on `WS1` and `WS2` being pointers to fast shared memory (SRAM).

<!-- okf:related:start -->

# 相关

- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax](../memory-sram/exp_attn_sinks_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax algorithm](../memory-sram/exp_flashattn_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations, softmax
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, softmax

<!-- okf:related:end -->
