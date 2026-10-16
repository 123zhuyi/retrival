---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: attention
dimension: "3.2"
title: 大序列 Attention 的分块计算与 Online Softmax 数值稳定
description: 针对大序列长度（seq_len≥1024）的 Attention，通过分块计算 QK^T、Online Softmax 和累加器复用，将内存占用从 O(S²) 降至 O(S)，提升 50%。
tags: [attention, memory_numerical, flash_attention, online_softmax, blocking, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/attention/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 大序列 Attention 的分块计算与 Online Softmax 数值稳定

## 概述

标准 Attention 需要计算并存储完整的 `S×S` 注意力分数矩阵，当序列长度 `S ≥ 1024` 时，HBM 带宽和容量都会成为瓶颈。该经验采用 **FlashAttention 风格的分块策略**：把 Q、K、V 都按 `BLOCK_SIZE` 分块，每次只加载一个 K/V 块到片上，计算局部 softmax 后增量更新输出累加器。

核心收益：

- **内存占用从 O(S²) 降至 O(S)**：无需保存完整 scores 矩阵；
- **数值稳定性**：每个分块先减去块内最大值再求 exp，避免指数溢出；
- **Online Softmax**：维护全局最大值 `m_i` 和全局 exp 和 `l_i`，在分块遍历中增量归一化；
- **V 复用**：输出累加器 `acc` 随分块迭代更新，避免重复加载 V。

典型收益在 `seq_len ≥ 1024`、`d_k ≥ 64`、`num_heads ≥ 8` 时达到 50%。

## 共同模式

### 1. Q/K/V 分块加载

每次只把一个 `BLOCK_SIZE × D` 的 K 块和 V 块加载到寄存器/L0，与当前 Q 块做 `tl.dot`。这样 HBM 访问量从 `O(S²)` 的 scores 矩阵降到 `O(S × BLOCK_SIZE)` 的 K/V 块多次加载。

### 2. Online Softmax 增量归一化

维护两个全局统计量：

- `m_i`：当前已处理分块中的全局最大值；
- `l_i`：当前已处理分块中 exp 值的累加和。

当新的分块到来时：

```python
m_ij = tl.maximum(m_i, tl.max(scores, axis=1, keepdims=True))
alpha = tl.exp(m_i - m_ij)
l_i = l_i * alpha + tl.sum(tl.exp(scores - m_ij), axis=1, keepdims=True)
acc = acc * alpha + tl.matmul(tl.exp(scores - m_ij), V)
m_i = m_ij
```

### 3. 块内数值稳定

每个分块的 scores 先减去该分块的最大值 `m_ij`，再计算 exp。这样所有 exp 值都 ≤1，避免了大数值导致的指数溢出。

### 4. 输出累加器复用 V

`acc` 在遍历 K/V 分块的过程中不断更新，最终只需一次 `acc / l_i` 归一化即可写回输出。V 块在计算当前分块时被加载一次，不需要为存储完整 scores 而反复读取。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_fused_attention_002` | 50% | 大序列 Attention 分块计算 + Online Softmax，将内存占用从 O(S²) 降至 O(S)，提升 50%。 |

## 代表 Kernel

`exp_fused_attention_002` 没有独立的 extracted kernel 文件，以下为 YAML 中的概念实现：

```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    B, H, S, D,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    b = pid // (H * (S // BLOCK_SIZE))
    h = (pid % (H * (S // BLOCK_SIZE))) // (S // BLOCK_SIZE)
    s_q = (pid % (S // BLOCK_SIZE)) * BLOCK_SIZE

    # 加载当前查询块 Q
    offs_m = tl.arange(0, BLOCK_SIZE)[:, None]
    offs_d = tl.arange(0, D)[None, :]
    q_ptrs = (
        Q_ptr + b * H * S * D + h * S * D
        + (s_q + offs_m) * D + offs_d
    )
    mask_q = (offs_m < S - s_q) & (offs_d < D)
    Q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)

    # 初始化全局统计量
    m_i = tl.full((BLOCK_SIZE, 1), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE, 1), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)

    # 分块处理 K/V
    for start_k in range(0, S, BLOCK_SIZE):
        k_ptrs = (
            K_ptr + b * H * S * D + h * S * D
            + (start_k + offs_m) * D + offs_d
        )
        v_ptrs = (
            V_ptr + b * H * S * D + h * S * D
            + (start_k + offs_m) * D + offs_d
        )
        mask_kv = (offs_m < S - start_k) & (offs_d < D)
        K = tl.load(k_ptrs, mask=mask_kv, other=0.0).to(tl.float32)
        V = tl.load(v_ptrs, mask=mask_kv, other=0.0).to(tl.float32)

        # QK^T / sqrt(D)
        scores = tl.matmul(Q, tl.trans(K)) * tl.rsqrt(tl.cast(D, tl.float32))
        mask_scores = tl.arange(0, BLOCK_SIZE)[None, :] < S - start_k
        scores = tl.where(mask_scores, scores, -float("inf"))

        # Online Softmax 更新
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1, keepdims=True))
        p = tl.exp(scores - m_ij)
        l_ij = tl.sum(p, axis=1, keepdims=True)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + tl.matmul(p, V)
        m_i = m_ij

    # 最终归一化并写回
    out_ptrs = (
        output_ptr + b * H * S * D + h * S * D
        + (s_q + offs_m) * D + offs_d
    )
    out_mask = (offs_m < S - s_q) & (offs_d < D)
    tl.store(out_ptrs, (acc / l_i).to(Q.dtype), mask=out_mask)
```

## 已知实例

- `exp_fused_attention_002` — Memory + Numerical Optimization（收益 50%）
  - 适用于 `seq_len ≥ 1024`、`d_k ≥ 64`、`num_heads ≥ 8` 的 Attention 场景。

## 关联知识

- 开发指南: [Attention 算子优化](../../reference/op-class-attention.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, attention, flash_attention
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, online_softmax
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention, flash_attention, online_softmax
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention, flash_attention, online_softmax
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 attention
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention, flash_attention, online_softmax
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention, online_softmax
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 attention, online_softmax
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention, flash_attention, online_softmax
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, flash_attention, online_softmax
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention, flash_attention
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, attention
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, attention, online_softmax
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3

<!-- okf:related:end -->
