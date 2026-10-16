---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）
description: 长序列注意力场景下，分块计算与在线 Softmax 避免物化完整评分矩阵，收益 50%
tags: [访存效率, mix, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 长序列注意力分块计算 + 在线 Softmax

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16，seq_len ≥1024、d_k ≥64、num_heads ≥8；失效边界：seq_len <1024 的短序列分块循环调度开销占比升高，收益收窄；num_heads <8 时 grid 并行度不足，核利用率成为瓶颈）

### 原则
- **分块计算**：将大序列长度矩阵按BLOCK_SIZE分块处理，避免存储完整的注意力分数矩阵，降低内存占用
- **在线Softmax**：增量式计算Softmax，维护全局最大值(m_i)和全局exp和(l_i)，无需存储完整scores矩阵
- **数值稳定**：分块内先减去块内最大值再计算exp，避免指数运算溢出，提升数值稳定性
- **内存复用**：累加器(acc)分块更新，减少重复加载V张量的开销

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def flash_attention_kernel(Q_ptr, K_ptr, V_ptr, output_ptr, B, H, S, D, blockSize: tl.constexpr=128):
    # Flash Attention优化实现：分块计算+在线Softmax
    pid = tl.program_id(0)
    b = pid // (H * (S // blockSize))
    h = (pid % (H * (S // blockSize))) // (S // blockSize)
    s_q = (pid % (S // blockSize)) * blockSize
    
    # 加载当前查询块Q
    q_ptr = Q_ptr + b * H * S * D + h * S * D + s_q * D + tl.arange(0, blockSize)[:, None] * D + tl.arange(0, D)[None, :]
    Q = tl.load(q_ptr, mask=tl.arange(0, blockSize)[:, None] < S - s_q and tl.arange(0, D)[None, :] < D, other=0.0)
    
    # 初始化全局统计量
    m_i = tl.full((blockSize, 1), -float('inf'), dtype=tl.float32)  # 全局最大值
    l_i = tl.full((blockSize, 1), 0.0, dtype=tl.float32)           # 全局exp和
    acc = tl.full((blockSize, D), 0.0, dtype=tl.float32)           # 输出累加器
    
    # 分块处理K/V（在线Softmax核心逻辑）
    for start_k in range(0, S, blockSize):
        # 1. 加载当前K/V块
        k_ptr = K_ptr + b * H * S * D + h * S * D + start_k * D + tl.arange(0, blockSize)[:, None] * D + tl.arange(0, D)[None, :]
        v_ptr = V_ptr + b * H * S * D + h * S * D + start_k * D + tl.arange(0, blockSize)[:, None] * D + tl.arange(0, D)[None, :]
        K = tl.load(k_ptr, mask=tl.arange(0, blockSize)[:, None] < S - start_k and tl.arange(0, D)[None, :] < D, other=0.0)
        V = tl.load(v_ptr, mask=tl.arange(0, blockSize)[:, None] < S - start_k and tl.arange(0, D)[None, :] < D, other=0.0)
        
        # 2. 计算当前块QK^T分数
        scores = tl.matmul(Q, tl.trans(K)) / tl.sqrt(tl.cast(D, tl.float32))
        load_mask = tl.arange(0, blockSize)[None, :] < S - start_k
        scores = tl.where(load_mask, scores, -float('inf'))
        
        # 3. 更新全局最大值
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1, keepdims=True))
        
        # 4. 计算当前块exp值（数值稳定化）
        scores = scores - m_ij
        p = tl.exp(scores)
        
        # 5. 更新全局exp和
        l_ij = tl.sum(p, axis=1, keepdims=True)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # 6. 更新输出累加器
        acc = acc * alpha + tl.matmul(p, V)
        
        # 7. 更新全局最大值
        m_i = m_ij
    
    # 最终归一化
    output = acc / l_i
    
    # 存储输出
    output_ptr = output_ptr + b * H * S * D + h * S * D + s_q * D + tl.arange(0, blockSize)[:, None] * D + tl.arange(0, D)[None, :]
    tl.store(output_ptr, output.to(Q.dtype), mask=tl.arange(0, blockSize)[:, None] < S - s_q and tl.arange(0, D)[None, :] < D)
```

**坏实践（反例）**：一次性物化完整 S×S 评分矩阵，且 exp 前不减最大值
```python
scores = tl.matmul(Q, tl.trans(K)) / sqrt(D)   # S×S 全量评分驻留内存
p = tl.exp(scores)                              # 未减 max，大 logits 直接溢出
out = tl.matmul(p / tl.sum(p, axis=1), V)
```
后果：内存占用 O(S²)，长序列下 GM 带宽与容量双双成为瓶颈；数值上 exp 溢出产生 inf/NaN。✅ 正解：按 BLOCK_SIZE 分块遍历 K/V，循环内用 m_i/l_i 增量维护在线 Softmax 统计量，评分矩阵永不物化，内存降至 O(S)，分块内先减块内最大值再 exp。

**已知实例**：attention §raw (`experiences/attention/exp_fused_attention_002.yaml`)
- 声称性能收益：50%
- 备注：适用于大序列长度（seq_len≥1024）的Attention计算，分块计算减少内存占用（从O(S²)降至O(S)），在线Softmax提升数值稳定性，910b2/910b3硬件上性能提升显著

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 attention
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 attention
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 attention
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 同属优化知识，主题相关
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 同属优化知识，主题相关
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](../compute-pipeline/exp_fused_attention_001.md) — 主题相关：共享 attention, mix
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](../compute-pipeline/exp_perf_010.md) — 主题相关：共享 mix
- 相关主题: [Compute Optimization: online softmax](../compute-pipeline/exp_sdpa_infer_ascend_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_online_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
