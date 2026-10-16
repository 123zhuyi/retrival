---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: attention
dimension: "6.3"
title: Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐
description: 针对 seq_len≤1024 的中小序列标准 Attention，覆盖 QK^T → 缩放 → Softmax → AV 的三步工作流、数值稳定技巧与 HEAD_DIM 维度对齐，是 Flash Attention 等高级优化的正确性基准，收益 15%。
tags: [attention, workflow_opt, standard_attention, softmax, causal_mask, fp32_accum, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/attention/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐

## 概述

标准 Attention 的计算流程为：

```
Scores = Q @ K.T / sqrt(d_k)
Scores = apply_causal_mask(Scores)
Attn   = softmax(Scores, dim=-1)
Out    = Attn @ V
```

对于中小序列（`seq_len ≤ 1024`），显式执行这三步既保证了**逻辑正确性**，也便于做**数值稳定性校验**，是 Flash Attention 等 fused 路径的基准版本。本卡唯一的 experience `exp_fused_attention_001` 强调：在优化之前，先把标准三步流程写对、写稳，收益 15%。

**核心 trade-off**：
- 三步分离会物化 `S×S` 的 score/attn 矩阵，内存复杂度 `O(S²)`，但在 S 较小时仍可接受；
- 分步实现调试简单，便于定位精度问题；
- 当 `S ≥ 1024` 时，应迁移到 Flash Attention/Online Softmax 以降低内存。

## 共同模式

### 1. 三步工作流：QK^T → Scale → Softmax → AV

**问题根因**：Attention 的数学定义就是四步操作：矩阵乘、缩放、softmax、矩阵乘。任何 fused 优化都必须等价于这四步结果。标准工作流把每一步显式化，便于验证。

**收益明显条件**：
- `seq_len ≤ 1024`；
- 需要与 PyTorch `scaled_dot_product_attention` 对齐；
- 作为 Flash Attention 的 baseline 或 fallback。

**实现要点**：
```python
# 步骤 1: Q @ K.T / sqrt(d_k)
scores = qk_matmul_kernel[grid_qk](Q, K, score_buf, scale, ...)
# 步骤 2: Softmax over last dim
attn = softmax_kernel[grid_softmax](score_buf, attn_buf, ...)
# 步骤 3: Attn @ V
out = av_matmul_kernel[grid_av](attn_buf, V, output, ...)
```
- 每个步骤一个 kernel，逻辑清晰；
- `score_buf` 和 `attn_buf` 用 `torch.empty` 分配；
- 每个 kernel 只负责单一操作，易于 autotune。

**常见坑**：
- 缩放因子用 `1/sqrt(d_k)` 而不是 `1/d_k`；
- causal mask 应用时机错误（应在 softmax 前）；
- 分步时 dtype 转换导致精度回退。

### 2. Softmax 数值稳定：减最大值 + fp32 累加

**问题根因**：Attention score 的绝对值可能很大（与 `d_k` 有关），直接 `exp` 会溢出。标准做法是先减最大值，再做 `exp` 和归一化。

**收益明显条件**：
- 任何使用 softmax 的 attention；
- `d_k` 较大或输入值范围宽；
- 需要与参考实现对齐精度。

**实现要点**：
```python
max_val = tl.max(scores, axis=1)
scores_shifted = scores - max_val[:, None]
exp_scores = tl.exp(scores_shifted)
sum_exp = tl.sum(exp_scores, axis=1)
attn = exp_scores / sum_exp[:, None]
```
- 先按行求 `max`；
- 所有元素减 `max`；
- `exp` 后求和并归一化；
- 中间结果用 fp32，最后 cast 到目标 dtype。

**常见坑**：
- 忘记减 `max` 导致 `exp` 溢出；
- 用 fp16 累加 `sum(exp)` 导致精度不足；
- causal mask 把 `-inf` 当 `0` 处理，softmax 分母错误。

### 3. Causal Mask：行位置 ≥ 列位置保留

**问题根因**：自回归 Attention 要求每个 token 只能 attend 到自己和之前的 token。mask 条件为 `q_pos >= kv_pos`。

**收益明显条件**：
- Decoder 自注意力；
- 训练/推理 causal 模型；
- 需要避免未来信息泄漏。

**实现要点**：
```python
q_idx = q_start + tl.arange(0, BLOCK_Q)[:, None]
kv_idx = kv_start + tl.arange(0, BLOCK_KV)[None, :]
scores = tl.where(q_idx >= kv_idx, scores, float('-inf'))
```
- 用绝对序列位置比较；
- `-inf` 经过 softmax 后变为 0；
- 注意 off-by-one：`q_pos == kv_pos` 应保留。

**常见坑**：
- 条件写成 `q_idx > kv_idx` 漏掉对角线；
- mask 与 padding 混淆；
- 非 causal 场景误加 causal mask。

### 4. HEAD_DIM 维度对齐：64 的倍数

**问题根因**：Ascend Cube 单元对 `HEAD_DIM` 有对齐要求，fp16/bf16 下通常需要是 64 的倍数。否则需要 padding，增加代码复杂度。

**收益明显条件**：
- 使用 fp16/bf16；
- `HEAD_DIM` 为 64/128/256；
- 需要走 Cube 矩阵乘。

**实现要点**：
- 在 host 侧断言 `HEAD_DIM % 64 == 0`；
- kernel 内 `HEAD_DIM` 用 `tl.constexpr`；
- 若模型 `HEAD_DIM` 非 64 倍数，host 侧先 padding 到 64 倍数。

**常见坑**：
- 把 `HEAD_DIM=80` 直接传入 kernel 而不 padding；
- `tl.dot` 输入 shape 不是 Cube 粒度倍数；
- 输出写回时未去掉 padding。

### 5. 输出张量创建：优先 `empty_like`

**问题根因**：中间 `score_buf`、`attn_buf` 和最终输出都会被 kernel 完整覆盖，使用 `torch.empty/empty_like` 可避免 `zeros` 的额外 fill 开销。

**收益明显条件**：
- 中间 buffer 会被完整覆盖；
- S 较大，buffer 元素数多；
- 热路径频繁分配。

**实现要点**：
```python
score_buf = torch.empty((B, H, S, S), dtype=torch.float32, device=Q.device)
attn_buf = torch.empty_like(score_buf)
output = torch.empty_like(Q)
```
- 中间 buffer 用 fp32 保证 softmax 精度；
- 最终输出 `empty_like(Q)` 继承 dtype/device；
- 不要用 `zeros_like` 创建输出。

**常见坑**：
- 中间 buffer 用 fp16，softmax 精度下降；
- 未初始化的 buffer 被下游读取；
- `score_buf` 的 shape 与 kernel 预期不一致。

### 6. 何时迁移到 Flash Attention

**问题根因**：标准三步 Attention 的内存复杂度是 `O(S²)`。当 `S` 较大时，中间 buffer 成为瓶颈。

**迁移条件**：
- `S > 1024` 或 batch×head 数量很大；
- `score_buf` 的显存占用超过预算；
- 需要 online softmax 分块计算。

**实现要点**：
- 在 host 侧根据 `S` 选择 kernel；
- S 小时用标准三步（调试友好）；
- S 大时用 Flash Attention/Online Softmax。

**常见坑**：
- 所有 shape 都走 Flash Attention，小 S 时 launch overhead 反而更差；
- 两套实现结果精度不一致，未做 tolerance 校验；
- 未处理 Flash Attention 的 causal mask 边界。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_fused_attention_001` | 15% | 中小序列（seq_len≤1024）标准 Attention：QK^T → 缩放 → Softmax → AV 三步工作流，保证逻辑正确性和基础数值稳定性，作为 Flash Attention 优化的基准版本。 |

## 代表 Kernel

选择唯一的 `exp_fused_attention_001` 作为代表。下面为合成的**标准 Attention 三步工作流**，包含 `qk_matmul`、`softmax_row`、`av_matmul` 三个 kernel，展示 workflow 的核心模式：

```python
import triton
import triton.language as tl
import torch
import math

# ---------- 步骤 1: Q @ K.T / sqrt(d_k) ----------
@triton.jit
def qk_matmul_kernel(
    q_ptr, k_ptr, score_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_srow, stride_scol,
    scale,
    S, HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr = 64,
    BLOCK_KV: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    kv_offs = tl.arange(0, BLOCK_KV)
    d_offs = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        q_ptr
        + pid_b * stride_qb + pid_h * stride_qh
        + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    )
    k_ptrs = (
        k_ptr
        + pid_b * stride_kb + pid_h * stride_kh
        + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
    )

    q = tl.load(q_ptrs, mask=q_offs[:, None] < S, other=0.0)
    acc = tl.zeros((BLOCK_Q, BLOCK_KV), dtype=tl.float32)
    for _ in range(0, tl.cdiv(HEAD_DIM, HEAD_DIM)):
        # 简化示例：一次性加载完整 HEAD_DIM
        k = tl.load(k_ptrs, mask=kv_offs[:, None] < S, other=0.0)
        acc += tl.dot(q, tl.trans(k))
        # HEAD_DIM 较小时可一次完成
        break

    score_ptrs = (
        score_ptr
        + pid_b * stride_sb + pid_h * stride_sh
        + q_offs[:, None] * stride_srow + kv_offs[None, :] * stride_scol
    )
    tl.store(score_ptrs, acc * scale, mask=(q_offs[:, None] < S) & (kv_offs[None, :] < S))


# ---------- 步骤 2: 按行 Softmax + Causal Mask ----------
@triton.jit
def softmax_row_kernel(
    score_ptr, attn_ptr,
    stride_sb, stride_sh, stride_srow, stride_scol,
    stride_ab, stride_ah, stride_arow, stride_acol,
    S,
    BLOCK_S: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_row = tl.program_id(2)

    row = pid_row * BLOCK_S + tl.arange(0, BLOCK_S)
    col = tl.arange(0, S)

    score_ptrs = (
        score_ptr
        + pid_b * stride_sb + pid_h * stride_sh
        + row[:, None] * stride_srow + col[None, :] * stride_scol
    )
    scores = tl.load(score_ptrs, mask=(row[:, None] < S) & (col[None, :] < S), other=float('-inf'))

    # causal mask: q_pos >= kv_pos
    q_pos = row[:, None]
    kv_pos = col[None, :]
    scores = tl.where(q_pos >= kv_pos, scores, float('-inf'))

    # 数值稳定 softmax
    max_val = tl.max(scores, axis=1)
    exp_scores = tl.exp(scores - max_val[:, None])
    sum_exp = tl.sum(exp_scores, axis=1)
    attn = exp_scores / sum_exp[:, None]

    attn_ptrs = (
        attn_ptr
        + pid_b * stride_ab + pid_h * stride_ah
        + row[:, None] * stride_arow + col[None, :] * stride_acol
    )
    tl.store(attn_ptrs, attn, mask=(row[:, None] < S) & (col[None, :] < S))


# ---------- 步骤 3: Attn @ V ----------
@triton.jit
def av_matmul_kernel(
    attn_ptr, v_ptr, out_ptr,
    stride_ab, stride_ah, stride_arow, stride_acol,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    S, HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr = 64,
    BLOCK_D: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, HEAD_DIM)

    attn_ptrs = (
        attn_ptr
        + pid_b * stride_ab + pid_h * stride_ah
        + q_offs[:, None] * stride_arow + tl.arange(0, S)[None, :] * stride_acol
    )
    v_ptrs = (
        v_ptr
        + pid_b * stride_vb + pid_h * stride_vh
        + tl.arange(0, S)[:, None] * stride_vs + d_offs[None, :] * stride_vd
    )

    attn = tl.load(attn_ptrs, mask=(q_offs[:, None] < S), other=0.0)
    v = tl.load(v_ptrs, mask=tl.arange(0, S)[:, None] < S, other=0.0)
    out = tl.dot(attn, v)

    out_ptrs = (
        out_ptr
        + pid_b * stride_ob + pid_h * stride_oh
        + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=(q_offs[:, None] < S) & (d_offs[None, :] < HEAD_DIM))


def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Q/K/V shape: [B, H, S, D]"""
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # 中间 buffer 用 fp32 保证 softmax 精度
    score_buf = torch.empty((B, H, S, S), dtype=torch.float32, device=Q.device)
    attn_buf = torch.empty_like(score_buf)
    output = torch.empty_like(Q)

    grid_qk = (B, H, triton.cdiv(S, 64))
    qk_matmul_kernel[grid_qk](
        Q, K, score_buf,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        score_buf.stride(0), score_buf.stride(1), score_buf.stride(2), score_buf.stride(3),
        scale, S, D,
    )

    grid_softmax = (B, H, triton.cdiv(S, 64))
    softmax_row_kernel[grid_softmax](
        score_buf, attn_buf,
        score_buf.stride(0), score_buf.stride(1), score_buf.stride(2), score_buf.stride(3),
        attn_buf.stride(0), attn_buf.stride(1), attn_buf.stride(2), attn_buf.stride(3),
        S,
    )

    grid_av = (B, H, triton.cdiv(S, 64))
    av_matmul_kernel[grid_av](
        attn_buf, V, output,
        attn_buf.stride(0), attn_buf.stride(1), attn_buf.stride(2), attn_buf.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        S, D,
    )

    return output
```

**关键实现细节**：
- 标准三步工作流：QK^T → scale → causal mask → softmax → AV；
- `score_buf`/`attn_buf` 使用 fp32，避免 softmax 精度损失；
- causal mask 在 softmax kernel 中按绝对位置应用；
- 输出 `torch.empty_like(Q)`，无初始化开销；
- 当 `S > 1024` 时，应迁移到 Flash Attention 以降低 `O(S²)` 内存。

## 已知实例

- `exp_fused_attention_001` — Workflow Opt（收益 15%）
  - 中小序列（seq_len≤1024）标准 Attention：QK^T → 缩放 → Softmax → AV 三步工作流，保证逻辑正确性和基础数值稳定性，是 Flash Attention 优化的基准版本。

## 关联知识

- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, attention, softmax
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, softmax
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention, softmax
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention, softmax
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 attention, softmax
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 主题相关：共享 softmax
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 softmax
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention, softmax
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention, softmax
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 attention, softmax
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention, softmax
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 fp32_accum
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, attention
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3, attention
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention

<!-- okf:related:end -->
