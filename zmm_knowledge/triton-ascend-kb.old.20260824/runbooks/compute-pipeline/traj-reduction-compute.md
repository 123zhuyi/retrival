---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: reduction
dimension: "2.1"
title: Reduction Compute 优化聚合
description: 聚合 3 条 reduction 算子在 Ascend 910B 上的 compute 优化经验，覆盖 SSM backward 序列维归约、Split-K FlashAttention 第二阶段聚合与 RMSNorm weight 梯度求和。
tags: [reduction, compute, ssm, flash_attention, rmsnorm, split_k, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/reduction/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Reduction Compute 优化聚合

## 概述

Reduction 算子看似简单，但在 Ascend 910B 上要做到高效，需要处理好**数据局部性、数值稳定性和并行度**之间的平衡。本卡聚合 3 条 reduction compute 经验，分别来自：
- SSM/SDA backward 中的序列维梯度归约；
- Split-K FlashAttention 的第二阶段部分结果聚合；
- RMSNorm backward 中 weight 梯度的行求和。

这些经验的共同点：归约维度通常很长，但每个输出元素只依赖部分输入，**调度策略和向量化加载**决定性能。

## 共同模式

### 1. Grid-Stride + 序列/块 Tiling

当归约维度很长时，不能指望一个 program 处理全部数据。典型做法：
- grid size 固定为核心数；
- 每个 program 通过 grid-stride 循环处理多个 tile；
- 每个 tile 内部做局部归约；
- 必要时再做跨 tile 的二次归约。

这适用于 SSM backward 中沿序列长度求 `d`、以及 Split-K 第二阶段聚合 partial sum。

### 2. 向量化 Block Loading

归约核的 load 必须对齐硬件向量宽度。优化要点：
- `BLOCK_SIZE` 对齐 128/256 bit；
- 使用 `tl.load` 一次性加载一个 block；
- 边界处用 mask，但尽量让内部循环无 mask。

### 3. 混合精度累加

所有涉及求和、乘积的归约都应在 fp32 累加，最后 downcast。这在 Split-K 聚合（需要合并多个 partial softmax 统计量）和 RMSNorm dW 求和中尤为关键。

### 4. Split-K 两阶段归约

FlashAttention Split-K 把输出 tile 的计算拆到多个 K 块上并行，第二阶段需要：
- 聚合多个 partial accumulator；
- 合并各自的 softmax max/sum 统计量（online softmax 风格）；
- 用数值稳定的 rescale 避免指数溢出。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_sda_bwd_reduce_001` | 25% | SSM/SDA backward 序列维归约：grid-stride、向量化 block load、混合精度累加 |
| `exp_reduce_split_001` | 20% | Split-K FlashAttention 第二阶段：数值稳定聚合、向量化、stride-aware 寻址 |
| `exp_backward_sum_001` | 25% | RMSNorm backward weight 梯度：向量化对齐、合并访问、block size 调优 |

## 代表 Kernel

以下为 `exp_sda_bwd_reduce_001` 对应的 kernel，体现了 **grid-stride + 序列 tiling + 向量化 load + 混合精度累加**：

```python
@triton.jit
def kernel_sda_bwd_d(
    dout_ptr, q_ptr, k_ptr, v_ptr, d_ptr,
    batch, n_heads, seq_len, d_head,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    # grid-stride 处理多个 row block
    for r_start in range(pid * BLOCK_R, seq_len, num_pid * BLOCK_R):
        r_offs = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offs < seq_len

        acc = tl.zeros((BLOCK_R, BLOCK_H), dtype=tl.float32)
        for h_start in range(0, d_head, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < d_head

            # 向量化加载 dout 和 q/k/v 的相关块
            g = tl.load(dout_ptr + r_offs[:, None] * stride_s + h_offs[None, :] * stride_d,
                        mask=r_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            # ... 与 q/k/v 计算得到 d 的局部贡献 ...
            acc += local_d

        tl.store(d_ptr + r_offs[:, None] * stride_s + tl.arange(0, BLOCK_H)[None, :] * stride_d,
                 acc.to(d_ptr.dtype.element_ty),
                 mask=r_mask[:, None])
```

关键实现细节：
- `BLOCK_R` 和 `BLOCK_H` 需根据 H 维度调优；
- 累加在 fp32 完成；
- grid-stride 循环覆盖整个序列长度。

## 已知实例

- `exp_sda_bwd_reduce_001` — Compute（收益 25%）
  - SSM/SDA backward 序列维梯度归约：grid-stride、混合精度累加、向量化 block loading、序列 tiling。
- `exp_reduce_split_001` — Compute（收益 20%）
  - Split-K FlashAttention 第二阶段：假设第一阶段已产生 partial accumulation 和统计量，做数值稳定的聚合。
- `exp_backward_sum_001` — Compute（收益 25%）
  - RMSNorm backward 中 scale 参数梯度：向量化对齐、合并访问、block size 调优。

## 关联知识

- 开发指南: [Reduction 算子优化](../../reference/op-class-reduction.md)
- 相关 runbook: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3, ssm
- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, reduction, split_k
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, reduction
- 对应指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 rmsnorm
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rmsnorm
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 rmsnorm
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 reduction
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 ssm
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 rmsnorm
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 reduction
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute, split_k
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute, rmsnorm
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 ssm
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 ssm
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, reduction
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, reduction
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 rmsnorm
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute, flash_attention
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute, ssm
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](traj-ssm-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, ssm

<!-- okf:related:end -->
