---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: softmax
dimension: "2.1"
title: Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定
description: 聚合 5 条 softmax/log_softmax 在 Ascend 910B 上的 compute 经验，覆盖 online softmax、two-pass 归约、向量化累加、FlashAttention 内联稳定计算与自动调参，收益 25%–35%。
tags: [softmax, compute, online_softmax, two_pass, vectorized_reduction, numerical_stability, flash_attention, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/softmax/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定

## 概述

Softmax/LogSoftmax 是**归约 + 逐元素** 的组合算子：先对某一维求 `max` 和 `sum(exp(x-max))`，再逐元素做 `exp(x-max) / sum` 或 `x - max - log(sum)`。Ascend 910B 上它的核心矛盾有三点：
1. **数值稳定性**：fp16/bf16 下 `exp(>11)` 就会溢出，必须 max-shift；
2. **归约开销**：水平 `tl.max`/`tl.sum` 在热循环中代价高；
3. **寄存器 vs 带宽**：一次把整行载入寄存器再做归约省带宽但寄存器压力大，分块 online 更新省寄存器但逻辑复杂。

本卡 5 条 experience 覆盖三种实现策略：
- **Online Softmax（单遍）**：边遍历边更新 running max 和 running sum，适合 FlashAttention 等 SRAM 受限场景；
- **Two-Pass（两遍）**：先归约出全局统计量，再逐元素计算，实现简单、寄存器压力小；
- **向量化累加**：把中间 max/sum 保持为向量，只在循环末尾做一次水平归约。

**核心 trade-off**：
- Online 减少内存遍历次数，但状态更新有数据依赖，对编译器调度要求更高；
- Two-Pass 可读性好、寄存器占用低，但会重复读取输入，纯内存 bound 场景吃亏；
- 向量化累加能隐藏水平归约，但需要足够大的 tile 来分摊尾归约开销。

## 共同模式

### 1. Online Softmax：单遍稳定归约

**问题根因**：传统 softmax 需要先求全局 max，再求全局 sum，至少两趟遍历。对 FlashAttention 这类无法把完整 N×N score 矩阵 materialize 到 HBM 的场景，两趟遍历不可接受。

**收益明显条件**：
- 归约维度 N 很大，或 tile 只能覆盖一部分 N；
- 输入在 SRAM/寄存器中只驻留一次（如 FlashAttention 的 qk tile）；
- 不希望重复读取 global memory。

**实现要点**：
维护两个 running 状态（fp32）：
```python
m_new = tl.maximum(m_old, x)
z_new = z_old * tl.exp(m_old - m_new) + tl.sum(tl.exp(x - m_new))
```
- `m` 是已见元素的最大值；
- `z` 是已见元素的 `sum(exp(x - m))`，按新 max 重新缩放；
- 最终 softmax = `exp(x - m_final) / z_final`。

**常见坑**：
- `m_old - m_new` 可能为 0 或负数，指数值在 `[0, 1]`，不会溢出；
- 初始 `m` 应设为 `-inf`，`z` 为 0；
- mask 掉的值（如 causal 或 padding）必须设为 `-inf`，不能是 0。

### 2. Two-Pass 算法：先归约后应用

**问题根因**：如果一次性把整行加载到寄存器，大 N 时寄存器 spill；如果分块 online 更新，代码复杂且数据依赖多。

**收益明显条件**：
- N 超过一个 tile，但可以分两次完整遍历；
- 寄存器/SRAM 不足以缓存整行；
- 对延迟不敏感、对实现正确性要求高的场景（如 log_softmax backward）。

**实现要点**：
- Pass 1：分块加载，fp32 累加得到 `max` 和 `sum_exp`（log_softmax 用 logsumexp）；
- Pass 2：再次分块加载，应用 `exp(x - max) / sum_exp` 或 `x - max - log(sum_exp)`。

**常见坑**：
- 会重复读取输入张量，内存 bound 场景需评估；
- log_softmax backward 需要先对 `out_grad` 求和，再与 `exp(out)` 相乘，天然适合 two-pass；
- 边界 mask 必须两趟一致。

### 3. 向量化累加：延迟水平归约

**问题根因**：热循环里反复调用 `tl.max`/`tl.sum` 会产生大量水平归约指令，Vector 单元利用率低。

**收益明显条件**：
- tile 维度（TILE_N）较大，可以先把每个 lane 的局部 max/sum 保留为向量；
- N 能被 TILE_N 大致整除，尾处理比例低。

**实现要点**：
- 循环内用 `tl.maximum(m_vec, x_vec)` 代替 `tl.max`；
- 用 `z_vec = z_vec * scale + tl.exp(x_vec - m_vec)` 更新；
- 循环结束后再 `tl.max(m_vec, axis=...)` 和 `tl.sum(z_vec, axis=...)`。

**常见坑**：
- 向量状态占用更多寄存器，tile 太大会 spill；
- 每次更新都需要按新 max 对历史 `z_vec` 做全局缩放，实现比亚标量 online 更复杂。

### 4. FlashAttention 内联 Softmax

**问题根因**：Attention score 的 N×N 矩阵太大，不能写到 HBM。必须在一个 kernel 内完成 `qk^T -> softmax -> pv`，并把 softmax 状态保存在寄存器/SRAM 中跨 KV tile 传播。

**收益明显条件**：
- 标准 FlashAttention / Split-K Attention 实现；
- KV 序列长，需要分多个 tile 计算。

**实现要点**：
- 每个 KV tile 计算局部 `qk`，做 online softmax 更新 running `m`/`l`；
- 输出 `O` 也要按 max 差值不断 rescale：
  ```python
  alpha = tl.exp(m_old - m_new)
  acc = acc * alpha + tl.dot(tl.exp(qk - m_new[:, None]), v)
  ```
- 最终 `O /= l[:, None]`。

**常见坑**：
- 跨 tile 的 rescale 必须在每个 tile 后立即执行，不能留到最后；
- `m` 和 `l` 必须是 fp32，且维度对齐 block 行；
- causal mask 要在 exp 之前把无效位置设为 `-inf`。

### 5. 数值稳定：max-shift + fp32 累加

**问题根因**：fp16 `exp(11.09)` 就溢出为 `Inf`；bf16 动态范围稍大但仍有限；低精度直接累加 `exp` 会累积舍入误差。

**收益明显条件**：
- 所有 fp16/bf16 softmax；
- attention score、MoE gating 等输入动态范围大的场景。

**实现要点**：
- 计算前先做 `x = x - tl.max(x, axis=dim)`；
- `max`、`sum`、`exp` 中间结果用 fp32；
- 最终输出按目标 dtype downcast。

**常见坑**：
- max 必须按 softmax 维度取，并与 `x` 广播一致；
- `tl.sum` 的 axis 参数要与 max 相同；
- log_softmax 的 `log(sum_exp)` 也要在 fp32 下算。

### 6. Tiling 与自动调参

**问题根因**：Softmax 的内存访问模式对 tile 形状敏感，固定 tile 无法覆盖多变的 (M, N, K)。

**收益明显条件**：
- M/N/K 变化范围大；
- 需要兼顾小 batch 和大 reduction dim。

**实现要点**：
- 用 `@triton.autotune` 针对 (M, N, K) 选择 BLOCK_M、BLOCK_N；
- 主循环去掉边界 mask（loop specialization），尾 tile 单独处理；
- log_softmax backward 等对读两次的数据可设置 `eviction_policy="evict_first"`，避免污染 L1。

**常见坑**：
- autotune key 必须包含真正变化的维度，否则选到次优配置；
- 主循环和尾循环分离时要确保索引不重叠、不遗漏。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_log_softmax_001` | 30% | LogSoftmax：two-pass + online 稳定 + N 维分块 tiling + autotune，适合大 N 归约。 |
| `exp_softmax_inner_002` | 30% | Softmax inner dim：向量化 online 累加，热循环内避免水平归约，配合 eviction_policy 优化 L1。 |
| `exp_log_softmax_backward_001` | 25% | LogSoftmax backward：two-pass 先求 out_grad 之和，再逐元素计算 in_grad，fp32 累加。 |
| `exp_softmax_non_inner_001` | 30% | 3D 张量非内维 Softmax：online/two-pass 启发式分发，向量化 (TILE_N, TILE_K) 访问。 |
| `exp_softmax_online_001` | 35% | FlashAttention 风格内联 online softmax：SRAM-resident workspace，fp32 状态跨 tile 传播。 |

## 代表 Kernel

选择 `exp_softmax_online_001` 作为代表（收益 35%），因为它体现了本组最核心的 **online softmax + FlashAttention 内联 + 数值稳定** 模式，且该模式在 attention 类算子中复用度最高。以下为根据该 experience 描述重构的代表示意：

```python
@triton.jit
def online_softmax_attention_tile(
    qk_ptr, v_ptr, out_ptr, lse_ptr,
    stride_qk_m, stride_qk_n,
    stride_v_n, stride_v_d,
    stride_o_m, stride_o_d,
    scale: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # 当前 Q 行/块的偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # running softmax 状态，必须 fp32
    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # 遍历 KV tile（online softmax 单遍）
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载 qk tile 并缩放
        qk = tl.load(
            qk_ptr + offs_m[:, None] * stride_qk_m + offs_n[None, :] * stride_qk_n,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=-float("inf"),
        ).to(tl.float32)
        qk *= scale

        # 局部 max / sum
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)

        # 更新 running sum 和输出累加器
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # p @ v，累加到 acc
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(p.to(tl.float32), v)

        m_i = m_new

    # 最终归一化并写回
    out = acc / l_i[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
        out.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
    )
    tl.store(lse_ptr + offs_m, m_i + tl.log(l_i), mask=offs_m < M)
```

**关键实现细节**：
- `m_i`/`l_i` 全程 fp32，避免低精度累加导致 logsumexp 漂移；
- 每次更新输出累加器 `acc` 前都要按 `alpha = exp(m_old - m_new)` 缩放，保证跨 tile 数值等价；
- mask 掉的 qk 位置设为 `-inf`，确保 `tl.max` 和 `exp` 不受影响；
- `lse`（log-sum-exp）在循环结束后一次性写出，可用于 backward rescale。

## 已知实例

- `exp_log_softmax_001` — Compute（收益 30%）
  - LogSoftmax：two-pass + online 稳定 + N 维分块 tiling + autotune，适合大 N 归约。
- `exp_softmax_inner_002` — Compute（收益 30%）
  - Softmax inner dim：向量化 online 累加，热循环内用 `tl.maximum` 代替 `tl.max`，配合 `eviction_policy="evict_first"`。
- `exp_log_softmax_backward_001` — Compute（收益 25%）
  - LogSoftmax backward：two-pass 先对 out_grad 求和得到 scale，再逐元素计算 in_grad，fp32 累加。
- `exp_softmax_non_inner_001` — Compute（收益 30%）
  - 3D 张量非内维 Softmax：online/two-pass 启发式分发，向量化 (TILE_N, TILE_K) 访问。
- `exp_softmax_online_001` — Compute（收益 35%）
  - FlashAttention 风格内联 online softmax：SRAM-resident workspace，fp32 running 状态跨 tile 传播，代表 kernel 见上。

## 关联知识

- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, flash_attention, softmax
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, numerical_stability, softmax
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, numerical_stability, online_softmax, softmax
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 flash_attention, online_softmax, softmax
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 flash_attention, online_softmax, softmax
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 主题相关：共享 softmax
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability, softmax
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention, numerical_stability, online_softmax
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute, online_softmax
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3, flash_attention, online_softmax
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, numerical_stability, softmax
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, softmax, two_pass
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 flash_attention
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute, online_softmax, two_pass
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute, numerical_stability
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute, flash_attention
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
