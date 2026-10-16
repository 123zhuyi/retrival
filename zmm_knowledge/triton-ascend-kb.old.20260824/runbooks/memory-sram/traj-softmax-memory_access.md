---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: softmax
dimension: "1.1"
title: Softmax Backward 内存访问优化聚合
description: 聚合 2 条 softmax backward 在 Ascend 910B 上的 memory access 经验，覆盖 two-pass 归约策略、自适应 tiling 与 inner/non-inner 维度的高效访问。
tags: [softmax, memory_access, backward, two_pass, reduction, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/softmax/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Softmax Backward 内存访问优化聚合

## 概述

Softmax backward 需要同时访问输出 `y` 和梯度 `dy`，并沿某一维度做归约。在 Ascend 910B 上，它的性能主要受内存访问模式影响：
- 归约维度是否连续，决定 load 是否能合并；
- 是否需要两趟遍历（one for sum，one for gradient），决定 HBM 带宽；
- tile 大小是否能放进 SRAM，同时覆盖足够数据以摊还开销。

本卡聚合 2 条 softmax backward 的 memory access 经验，分别针对 **inner 维度** 和 **non-inner 维度** 的归约场景，收益均为 30%。

## 共同模式

### 1. Two-Pass 归约策略

Softmax backward 的公式为：`dx = y * (dy - sum(dy * y))`。它需要先计算 `sum(dy * y)`，再用这个结果计算最终梯度。Two-pass 策略：
- 第一趟：分块加载 `y` 和 `dy`，累加 `sum(dy * y)`；
- 第二趟：再次加载 `y` 和 `dy`，应用公式并 store `dx`。

为什么不用 one-pass？因为 one-pass 需要同时保存所有中间结果，寄存器压力大，且 non-inner 维度场景下数据布局不利于合并访问。Two-pass 用多一次读取换取更简单的寄存器管理和更好的向量化。

### 2. 自适应 Tiling 与启发式

Softmax 的归约维度可能是 inner（最后一维）或 non-inner（其他维）。两种场景需要不同的 tiling：
- **inner 维度**：数据连续，可以用大 tile 一次加载整行；
- **non-inner 维度**：数据跨 stride，需要 2D tiling 并仔细处理 mask。

通过 `ONE_TILE_PER_CTA` 等启发式，可以为小 N 场景提供快速路径：当一行数据能放进单个 tile 时，跳过复杂的分块循环。

### 3. 显式 Cache 控制

在某些实现中，通过显式 cache eviction policy 或 prefetch hint，可以：
- 避免 softmax score 被不必要地保留在 cache；
- 让下一趟遍历的数据提前进入 cache；
- 减少长序列场景下的 cache thrashing。

### 4. 高精度累加

Softmax backward 中的 `sum(dy * y)` 如果在大维度上用 fp16 累加，误差会显著放大。所有实现都应在 fp32 上累加，最后再把 `dx` downcast。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_softmax_backward_001` | 30% | Softmax backward inner 维度：two-pass、显式 cache 策略、自适应 tiling |
| `exp_softmax_backward_non_inner_001` | 30% | Softmax backward non-inner 维度：two-pass、自适应 tiling、高精度累加、合并访问 |

## 代表 Kernel

以下为 `exp_softmax_backward_non_inner_001` 对应的 kernel，体现了 **non-inner 维度 two-pass + 2D tiling + 高精度累加**：

```python
@triton.jit
def softmax_backward_kernel_non_inner(
    y_ptr, dy_ptr, dx_ptr,
    M, N,
    stride_y_m, stride_y_n,
    stride_dy_m, stride_dy_n,
    stride_dx_m, stride_dx_n,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pass 1: 计算 sum(dy * y) 沿 N 维
    acc = tl.zeros((TILE_M,), dtype=tl.float32)
    for n_start in range(0, N, TILE_N):
        ns = n_start + offs_n
        m = mask_n & (ns < N)
        y = tl.load(y_ptr + offs_m[:, None] * stride_y_m + ns[None, :] * stride_y_n,
                    mask=mask_m[:, None] & m[None, :], other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dy_m + ns[None, :] * stride_dy_n,
                     mask=mask_m[:, None] & m[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(y * dy, axis=1)

    # Pass 2: 计算 dx = y * (dy - acc)
    for n_start in range(0, N, TILE_N):
        ns = n_start + offs_n
        m = mask_n & (ns < N)
        y = tl.load(y_ptr + offs_m[:, None] * stride_y_m + ns[None, :] * stride_y_n,
                    mask=mask_m[:, None] & m[None, :], other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dy_m + ns[None, :] * stride_dy_n,
                     mask=mask_m[:, None] & m[None, :], other=0.0).to(tl.float32)
        dx = y * (dy - acc[:, None])
        tl.store(dx_ptr + offs_m[:, None] * stride_dx_m + ns[None, :] * stride_dx_n,
                 dx.to(dx_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & m[None, :])
```

关键实现细节：
- 2D grid：`(M / TILE_M, N / TILE_N)`；
- 沿 N 维做 two-pass，每次加载 2D tile；
- `acc` 是 fp32 向量，每行一个累加值；
- mask 同时处理 M 和 N 边界。

## 已知实例

- `exp_softmax_backward_001` — Memory Access（收益 30%）
  - Softmax backward inner 维度：two-pass 归约策略、显式 cache eviction policy、自适应 tiling 与启发式，`ONE_TILE_PER_CTA` 小 N 快速路径。
- `exp_softmax_backward_non_inner_001` — Memory Access（收益 30%）
  - Softmax backward non-inner 维度：two-pass 归约、自适应 tiling、fp32 高精度累加、合并访问处理 N 非倍数边界，代表 kernel 见上。

## 关联知识

- 开发指南: [Softmax 算子优化](../../reference/op-class-softmax.md)
- 相关 runbook: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md)
- 相关 runbook: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, softmax
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, softmax
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, reduction
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, reduction, softmax
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 backward, softmax
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 softmax
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 主题相关：共享 softmax
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 reduction
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 reduction
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, two_pass
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Loss 计算优化聚合](../compute-pipeline/traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Normalization Scheduling 优化聚合](../compute-pipeline/traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, reduction
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, softmax, two_pass
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, backward, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, reduction
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
