---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: loss
dimension: "2.1"
title: Loss 计算优化聚合
description: 聚合 8 条 loss 算子在 Ascend 910B 上的 compute 优化经验，覆盖 Online Softmax、Two-Pass 稳定计算、大词汇表向量化、label smoothing 与 tensor parallelism。
tags: [loss, compute, cross_entropy, online_softmax, two_pass, tensor_parallelism, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/loss/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Loss 计算优化聚合

## 概述

Loss 层（尤其是 Cross Entropy、Entropy、GRPO loss）在大模型训练中面临两个核心挑战：**数值稳定性**和**大词汇表下的内存带宽**。当类别数 C 达到 32k 甚至 128k 时，把整个 softmax 矩阵驻留寄存器是不现实的，而分两次遍历 logits 又会在 HBM 上产生额外读写。本卡聚合 8 条 loss 类别的 compute 经验，收益 25%–60%，覆盖 Online Softmax、Two-Pass 算法、label smoothing、ignore_index、z_loss 等常见变体。

这些经验的共同目标：
- 用 O(C) 内存完成 softmax + loss 计算；
- 在 fp32 上累加，避免大 C 下的指数爆炸；
- 把 label smoothing、weighting、TP all-reduce 等逻辑融合进主循环。

## 共同模式

### 1. Online Softmax / LogSumExp 稳定计算

直接计算 `log(sum(exp(x)))` 会在大 C 下溢出。标准做法是维护 running max 和 running sum：
- 第一遍：求 `max` 和 `sum_exp(x - max)`；
- 第二遍：用 `log_sum_exp = max + log(sum)` 计算 loss 和梯度。

所有经验都把中间累加放在 fp32，并且只在最后 downcast。这是大词汇表 loss 的数值安全基线。

### 2. Two-Pass 算法平衡寄存器与带宽

Two-Pass 先在第一遍计算 LogSumExp，第二遍再计算 loss/梯度。相比 one-pass 需要同时保存中间激活，two-pass 的 trade-off 是：
- 多一次 logits 读取；
- 但省下了存储完整 [N, C] softmax 矩阵的内存；
- 适合 C 极大、寄存器/SRAM 存不下完整激活的场景。

优化重点：把两趟遍历的 block size 对齐到硬件 burst length（如 64/128 元素），让每次 load 都完全合并。

### 3. 大词汇表向量化与 block tiling

当 C 很大时，单个 program 需要遍历整个 C 维度。优化方式：
- 用 `tl.arange` 一次性加载 BLOCK_SIZE_C 个元素；
- BLOCK_SIZE_C 对齐硬件向量宽度（fp16/bf16 下 64/128 元素对应 128/256 字节）；
- 通过 grid-stride 或 persistent kernel 让一个 program 处理多个 sample。

### 4. label smoothing / ignore_index / z_loss 融合

这些变体都会在 softmax 计算前后引入额外逻辑。优化关键是把它们融入同一遍遍历：
- `ignore_index`：用 mask 跳过无效 label；
- `label_smoothing`：在计算梯度时直接加上平滑项，不单独存储 softmax；
- `z_loss`：在 forward 同时累加 logsumexp 的平方项；
- `weighting`：按 per-token weight 缩放梯度。

这样可以避免为每个变体单独写 kernel 或多次读写 logits。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_crossentropy_backward_001` | 60% | Cross Entropy backward：grid-stride、向量化 block load、two-pass、原地更新 |
| `exp_celoss_smooth_001` | 30% | CE loss + label smoothing，two-pass，大 C 场景 |
| `exp_celoss_001` | 35% | CE loss：two-pass 平衡寄存器与带宽，online LogSumExp |
| `exp_crossentropy_001` | 40% | Cross Entropy：online softmax，forward/backward 融合，大 block 向量化 |
| `exp_fused_entropy_001` | 60% | 融合 entropy，online softmax，支持 Tensor Parallelism |
| `exp_celoss_bwd_001` | 35% | CE loss backward：two-pass、稳定 LSE、label smoothing/weighting 融合 |
| `exp_celoss_smooth_bwd_001` | 30% | CE loss + smoothing backward：two-pass、向量化 tiling |
| `exp_entropy_epilogue_001` | 25% | Entropy epilogue：稳定 LSE、原子聚合、多 reduction 模式 |

## 代表 Kernel

以下为 `exp_crossentropy_001` 对应的 kernel，收益 40%，体现了 **online softmax + forward/backward 融合 + 大词汇表向量化**：

```python
@triton.jit
def _cross_entropy_kernel(
    logits_ptr, labels_ptr, losses_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # 第一遍：online softmax 计算 max 和 sum_exp
    col_offsets = tl.arange(0, BLOCK_SIZE)
    logits_ptrs = logits_ptr + row_idx * stride_row + col_offsets

    max_logit = tl.full([BLOCK_SIZE], value=-float('inf'), dtype=tl.float32)
    sum_exp = tl.full([BLOCK_SIZE], value=0.0, dtype=tl.float32)

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + col_offsets
        mask = cols < n_cols
        logits = tl.load(logits_ptrs + col_start, mask=mask, other=-float('inf')).to(tl.float32)

        new_max = tl.maximum(max_logit, logits)
        sum_exp = sum_exp * tl.exp(max_logit - new_max) + tl.exp(logits - new_max)
        max_logit = new_max

    # 跨 block 归约
    max_logit = tl.max(max_logit, axis=0)
    sum_exp = tl.sum(sum_exp * tl.exp(max_logit - max_logit), axis=0)
    log_sum_exp = max_logit + tl.log(sum_exp)

    # 第二遍：计算 loss
    label = tl.load(labels_ptr + row_idx)
    target_logit = tl.load(logits_ptr + row_idx * stride_row + label).to(tl.float32)
    loss = log_sum_exp - target_logit
    tl.store(losses_ptr + row_idx, loss)
```

关键实现细节：
- 使用 online softmax 算法，只需 O(1) 额外内存；
- 所有累加在 fp32 完成；
- BLOCK_SIZE 对齐硬件向量宽度，保证合并访问；
- forward/backward 可共享同一 LogSumExp 值，减少重复计算。

## 已知实例

- `exp_crossentropy_backward_001` — Compute（收益 60%）
  - Cross Entropy backward：grid-stride 调度、向量化 block load、two-pass 算法、操作融合、原地内存更新。
- `exp_celoss_smooth_001` — Compute（收益 30%）
  - CE loss + label smoothing，two-pass 遍历 logits，大 C 场景。
- `exp_celoss_001` — Compute（收益 35%）
  - CE loss：two-pass 平衡寄存器压力与内存带宽，online LogSumExp。
- `exp_crossentropy_001` — Compute（收益 40%）
  - Cross Entropy：online softmax、forward/backward 融合、大 block 向量化，代表 kernel 见上。
- `exp_fused_entropy_001` — Compute（收益 60%）
  - 融合 entropy 计算，online softmax，支持 Tensor Parallelism 场景。
- `exp_celoss_bwd_001` — Compute（收益 35%）
  - CE loss backward：two-pass、稳定 LSE、label smoothing 与 weighting 融合。
- `exp_celoss_smooth_bwd_001` — Compute（收益 30%）
  - CE loss + smoothing backward：two-pass、向量化 tiling、动态 shape 编译优化。
- `exp_entropy_epilogue_001` — Compute（收益 25%）
  - Entropy epilogue：稳定 LSE、向量化 block load、原子聚合、支持 store/sum/mean 多种 reduction。

## 关联知识

- 开发指南: [Loss 算子优化](../../reference/op-class-loss.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 910b2, 910b3, cross_entropy, loss
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute, online_softmax
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](../portability/traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute, online_softmax, two_pass
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
