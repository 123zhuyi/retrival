---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: '流程优化: 标准 score-归一化-加权三步流程'
description: 'attention 场景下，流程优化: 标准 score-归一化-加权三步流程，收益 15%'
tags: [多核并行与负载均衡, mix, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 流程优化: 标准 score-归一化-加权三步流程

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：本条是不分块的基准实现，seq_len > 1024 或 d_k 较大时完整 score 矩阵放不下片上，必须迁到分块/online 方案；仅适合中小序列（≤1024）作为功能正确性基线）

### 原则
- **计算流程**：严格遵循 score 矩阵乘（inputA·inputB^T）→ Softmax 归一化 → 加权求和（×inputC）三步标准流程，保证计算逻辑正确性
- **数值稳定**：score 结果除以 sqrt(d_k) 缩放，避免分数值过大导致 Softmax 数值溢出
- **维度对齐**：确保 inputA/inputB/inputC 张量维度匹配，均为 [B, H, S, D]，输出为 [B, H, S, D]

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def standard_attention_kernel(Q_ptr, K_ptr, V_ptr, output_ptr, B, H, S, D, blockSize: tl.constexpr=128):
    # 标准Attention计算流程实现
    pid = tl.program_id(0)
    # 解析batch/head/seq维度索引
    b = pid // (H * (S // blockSize))
    h = (pid % (H * (S // blockSize))) // (S // blockSize)
    s = (pid % (S // blockSize)) * blockSize
    
    # 1. QK^T计算：scores = Q @ K^T / sqrt(D)
    q_ptr = Q_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, blockSize)[:, None] * D + tl.arange(0, D)[None, :]
    k_ptr = K_ptr + b * H * S * D + h * S * D + tl.arange(0, S)[:, None] * D + tl.arange(0, D)[None, :]
    Q = tl.load(q_ptr, mask=tl.arange(0, blockSize)[:, None] < S - s and tl.arange(0, D)[None, :] < D, other=0.0)
    K = tl.load(k_ptr, mask=tl.arange(0, S)[:, None] < S and tl.arange(0, D)[None, :] < D, other=0.0)
    scores = tl.matmul(Q, tl.trans(K)) / tl.sqrt(tl.cast(D, tl.float32))
    
    # 2. Softmax归一化：attn_weights = softmax(scores)
    max_score = tl.max(scores, axis=1, keepdims=True)
    stable_scores = scores - max_score
    exp_scores = tl.exp(stable_scores)
    sum_exp = tl.sum(exp_scores, axis=1, keepdims=True)
    attn_weights = exp_scores / sum_exp
    
    # 3. 加权求和：output = attn_weights @ V
    v_ptr = V_ptr + b * H * S * D + h * S * D + tl.arange(0, S)[:, None] * D + tl.arange(0, D)[None, :]
    V = tl.load(v_ptr, mask=tl.arange(0, S)[:, None] < S and tl.arange(0, D)[None, :] < D, other=0.0)
    output = tl.matmul(attn_weights, V)
    
    # 存储输出
    output_ptr = output_ptr + b * H * S * D + h * S * D + s * D + tl.arange(0, blockSize)[:, None] * D + tl.arange(0, D)[None, :]
    tl.store(output_ptr, output, mask=tl.arange(0, blockSize)[:, None] < S - s and tl.arange(0, D)[None, :] < D)
```

**坏实践（反例）**：省略数值稳定步骤、对未减 max 的原始 score 直接 `tl.exp` → 大 logits 下 exp 上溢，softmax 结果 NaN → ✅ 正解即本条：先按行减 max 再 exp 求和归一（并先乘 1/sqrt(d_k) 控制量级）。

**已知实例**：attention §raw (`experiences/attention/exp_fused_attention_001.yaml`)
- 声称性能收益：15%
- 备注：适用于中小序列长度（seq_len≤1024）的标准 attention 计算，保证逻辑正确性和基础数值稳定性，是 attention 类 kernel 分块/在线化优化的基准版本

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention
- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 attention
- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 主题相关：共享 attention, mix
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 mix, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡

<!-- okf:related:end -->
