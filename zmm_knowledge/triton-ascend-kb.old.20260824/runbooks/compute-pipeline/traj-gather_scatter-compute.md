---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: gather_scatter
dimension: "2.1"
title: Scatter Weight Gradient 向量化计算优化
description: 针对 gather/scatter 类 weight gradient 计算，将标量循环改写为 batched 张量规约，配合对齐的 BLOCK_X 与早期 fp32 提升，在 910B 上提升 35%。
tags: [gather_scatter, compute, scatter_wgrad, vectorization, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/gather_scatter/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Scatter Weight Gradient 向量化计算优化

## 概述

在 MoE、索引采样等场景下，weight gradient 需要通过 `indices` 把每个 token 的梯度 scatter 到对应的 weight 位置。原始实现常用 Python 级标量循环逐个索引计算点积，导致 ALU 利用率低、vector pipe 空闲。`_scatter_wgrad_kernel` 展示了如何把这一过程转化为 **batched 张量规约**，一次性处理 `SUB_BLOCK_SIZE` 个索引的向量点积。

该模式的关键权衡：

- 用 batched `tl.sum(loaded_x * loaded_grads, axis=1)` 替代 `for i in range(SUB_BLOCK_SIZE)`，提升 ALU 利用率；
- `BLOCK_X` 必须与缓存线宽（32/64 字节）对齐，保证 global memory 事务合并；
- 在加载后立即 `.to(tl.float32)`，减少混合精度累加循环里的显式转换指令；
- 如果多个 token 映射到同一个 `wgrad` 索引，需要额外加 `tl.atomic_add`，否则会有写竞争。

典型收益在 `INDICES_LENGTH ≥ 256`、`NUM_COLUMNS ≥ 64` 时达到 35%。

## 共同模式

### 1. 标量循环 → Batched 张量规约

原始 kernel 用 `for i in range(0, SUB_BLOCK_SIZE)` 逐个索引调用 `tl.get_element`、`tl.extract_slice` 和 `tl.sum`。这会把向量数据拆成标量操作。优化方向是一次性加载 `SUB_BLOCK_SIZE × BLOCK_X` 的 `x` tile 和 `SUB_BLOCK_SIZE × BLOCK_X` 的 `grads` tile，然后沿列做 `tl.sum(x * grads, axis=1)`，得到 `SUB_BLOCK_SIZE` 个点积结果。

### 2. 索引与数据 tile 一起加载

`cur_indices = tl.load(indices + idx_offsets)` 与 `cur_x = tl.load(x + idx_offsets[:, None] * NUM_COLUMNS + ...)` 同步加载，保证索引和对应数据在同一轮循环内可用，避免索引加载与数据加载之间的依赖气泡。

### 3. BLOCK_X 与缓存线对齐

`col_offsets = tl.arange(0, BLOCK_X)` 决定了每次 load 的宽度。`BLOCK_X` 取 64/128 等硬件向量宽度倍数时，global memory 事务可以完全合并；否则尾部会触发标量或部分向量事务。

### 4. 早期 fp32 提升与输出 atomic

`cur_x` 和 `grads` 在加载后立刻提升到 `tl.float32`，避免在乘加循环里反复转换。最终写回 `wgrad` 时如果存在索引冲突，需要把 `tl.store` 替换为 `tl.atomic_add`；该经验默认索引唯一，因此直接用 store。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_scatter_wgrad_001` | 35% | 将 scatter weight gradient 中的标量循环改写为 batched 张量规约，并对齐 BLOCK_X，在 910B 上提升 35%。 |

## 代表 Kernel

以下为 `exp_scatter_wgrad_001` 对应的 kernel：

```python
@triton.jit
def _scatter_wgrad_kernel(
    x,
    grads,
    wgrad,
    indices,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)

    col_offsets = tl.arange(0, BLOCK_X)
    col_mask = col_offsets < NUM_COLUMNS
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if SUB_BLOCK_SIZE > 1:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            cur_x = tl.load(
                x + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                idx_mask[:, None] & col_mask[None, :],
                other=0,
            ).to(tl.float32)
            cur_grads = tl.load(
                grads + (cur_indices // TOP_K)[:, None] * NUM_COLUMNS + col_offsets[None, :],
                idx_mask[:, None] & col_mask[None, :],
                other=0,
            ).to(tl.float32)
            out = tl.sum(cur_x * cur_grads, axis=1)
            tl.store(wgrad + cur_indices, out.to(wgrad.dtype.element_ty), mask=idx_mask)

        elif idx_offset < idx_end:
            idx = tl.load(indices + idx_offset)
            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                data = tl.load(x + idx * NUM_COLUMNS + col_offsets, col_mask, other=0)
                grad = tl.load(grads + (idx // TOP_K) * NUM_COLUMNS + col_offsets, col_mask, other=0)
                acc += data.to(tl.float32) * grad.to(tl.float32)
            acc = tl.sum(acc)
            tl.store(wgrad + idx, acc.to(wgrad.dtype.element_ty))
```

## 已知实例

- `exp_scatter_wgrad_001` — Compute Optimization（收益 35%）
  - 适用于 scatter weight gradient 场景，关键参数 `INDICES_LENGTH ≥ 256`、`NUM_COLUMNS ≥ 64`、`SUB_BLOCK_SIZE ≥ 1`。
  - 对应 kernel: `raw_experiences/extracted_kernels/016-_scatter_wgrad_kernel.py`

## 关联知识

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, vectorization
- 对应指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, gather_scatter
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 同属优化知识，主题相关
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](../../ops/moe-init-routing-v2-grad.md) — 同属优化知识，主题相关
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](../../ops/moe-token-permute-with-ep-grad.md) — 同属优化知识，主题相关
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../memory-sram/parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, gather_scatter, vectorization
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, gather_scatter
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute, vectorization
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute, vectorization
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
