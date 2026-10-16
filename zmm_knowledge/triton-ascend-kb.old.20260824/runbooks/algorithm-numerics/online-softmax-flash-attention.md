---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.3"
title: Online Softmax / Flash Attention 单遍稳定计算
description: Flash Attention 用 online softmax 在 K/V 循环中维护 running max/sum/acc，将 attention 内存从 O(S²) 降到 O(S)，同时避免 exp 溢出。
tags: [online_softmax, flash_attention, running_max, running_sum, numerical_stability]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-24 Online Softmax / Flash Attention 单遍稳定计算

【优化维度】：数值｜数值稳定
【泛化层级】：Triton通用（迁移条件：softmax、attention、log-sum-exp 等需要跨块归约的场景；失效边界：序列很短且可一次装入 UB 的简化场景）

<原则：标准 softmax 需要 O(S²) 的 attention 矩阵，大序列时无法放入 UB。Online softmax 在 K/V 循环中维护 running max、running sum 和 running acc，每块只加载当前 K/V block，将内存复杂度降到 O(S)，同时通过减最大值保证 exp 不溢出。>

**Online Softmax 更新公式**：

```
m_ij = max(m_i, max(scores_block))        # 新的 running max
p = exp(scores_block - m_ij)              # 稳定 exp
l_ij = sum(p)                             # 局部 denominator
alpha = exp(m_i - m_ij)                   # 旧和修正因子
l_i = l_i * alpha + l_ij                 # 修正 running denominator
acc = acc * alpha + (p @ V_block)        # 修正 running numerator
m_i = m_ij                               # 更新 running max
```

最终输出：`output = acc / l_i`

**内存对比**：

| 方法 | Attention 内存 | S=4096, fp16 |
|---|---|---|
| 标准 softmax | O(S²) | 32MB |
| Online softmax | O(S) | ~48KB |

**Triton 模式**：

```python
@triton.jit
def flash_attention_kernel(Q, K, V, Out, ...):
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for k_start in range(0, seq_len, BLOCK_N):
        k = tl.load(K_ptr + k_start * d + k_offs, mask=k_mask)
        scores = tl.dot(q, tl.trans(k)) * scale

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + l_ij
        v = tl.load(V_ptr + k_start * d + v_offs, mask=v_mask)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    output = acc / l_i[:, None]
    tl.store(Out_ptr + ..., output.to(tl.float16), ...)
```

**关键注意点**：

- `m_i`、`l_i`、`acc` 必须全程 fp32。
- `p` 与 `v` 的 `tl.dot` 前需 cast 到合适精度（如 fp16）。
- causal mask 可在 `scores` 处通过 `m_ij` 前的 mask 实现。

**坏实践（反例）**：
```python
scores = tl.dot(q, tl.trans(k))
probs = tl.softmax(scores)  # 物化整个 S×S attention 矩阵
```
后果：大序列 UB overflow，且 exp 可能溢出。✅ 正解即 online softmax 单遍计算。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_fused_attention_002` — 内存优化+数值稳定优化 （收益 50%）
  - 适用于大序列长度（seq_len≥1024）的Attention计算，分块计算减少内存占用（从O(S²)降至O(S)），在线Softmax提升数值稳定性，910b2/910b3硬件上性能提升显著

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 flash_attention
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 online_softmax
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 online_softmax
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 flash_attention
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 numerical_stability
- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 numerical_stability
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 numerical_stability, online_softmax
- 相关主题: [Flash Attention Causal Skeleton](../../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 numerical_stability
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 flash_attention, numerical_stability, online_softmax
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 flash_attention
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](traj-attention-compute.md) — 主题相关：共享 online_softmax
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](traj-attention-memory_numerical.md) — 主题相关：共享 flash_attention, online_softmax
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](traj-general_stability-numerical_stability.md) — 主题相关：共享 numerical_stability

<!-- okf:related:end -->
