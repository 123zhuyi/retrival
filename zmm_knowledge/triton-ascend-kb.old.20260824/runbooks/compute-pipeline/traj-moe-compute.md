---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: moe
dimension: "2.1"
title: MoE 稀疏 Weight Gradient 计算优化
description: 针对 MoE 层 sparse weight gradient 的向量化点积实现，通过 ILP 批处理、fp32 累加和地址对齐，在 910B 上提升 25%。
tags: [moe, compute, sparse_gradient, vectorization, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/moe/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# MoE 稀疏 Weight Gradient 计算优化

## 概述

MoE 层的 weight gradient 具有典型的稀疏性：每个 token 只路由到 Top-K 个专家，每个专家只对其接收到的 token 计算梯度。`_binned_copy_wgrad_npu` 通过按专家分桶的稀疏点积，把每个 `(token, expert)` 对的梯度规约到对应的 expert weight 上。

该实现的核心权衡是 **ILP 与寄存器压力** 之间取平衡：

- 每个 program 处理 `BLOCK_SIZE` 个 token，提高指令级并行，掩盖内存延迟；
- 内层循环以 `BLOCK_X` 为步长做向量点积，最大化向量寄存器利用率；
- 中间累加使用 `tl.float32`，避免 fp16/bf16 大向量规约时的精度损失；
- 通过 `tl.multiple_of` 告知编译器地址对齐关系，帮助生成高效的合并 load/store。

收益在 `NUM_COLUMNS ≥ 128`、`BLOCK_X ≥ 64` 时较明显，适合 MoE 路由维度较大的场景。

## 共同模式

### 1. 专家分桶 + Top-K 索引反向映射

MoE 前向时每个 token 选择 Top-K 专家；反向时需要把梯度写回这些专家对应的 weight。实现上先把 token 按专家排序并记录 `bins` 偏移，kernel 里通过 `indices` 把分桶后的 token 映射回原始 token，从而把随机 scatter 转化为专家内的连续规约。

### 2. ILP 批处理：每个 program 负责多个 token

`BLOCK_SIZE > 1` 时，单个 program 顺序处理多个 token。由于不同 token 的 `x` 和 `grad` 地址不同但计算独立，编译器可以重叠 load/use，隐藏 HBM 延迟。注意 `BLOCK_SIZE` 过大会增加寄存器压力，通常与 expert 接收到的 token 数对齐选取。

### 3. 向量宽度对齐的点积内层循环

内层循环 `for _ in range(tl.cdiv(NUM_COLUMNS, BLOCK_X))` 以 `BLOCK_X` 为步长加载 `x` 和 `grad`，做向量乘加后 `tl.sum` 规约。`BLOCK_X` 应取硬件向量宽度的整数倍（如 64/128），否则尾部处理会引入标量路径。

### 4. fp32 中间累加 + 地址对齐提示

输入是 fp16/bf16，但 `acc` 用 `tl.float32`；最终 `tl.sum(acc).to(wgrad.dtype.element_ty)` 再写回。`tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)` 与 `tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)` 配合，让编译器推断出连续且对齐的访存模式。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_moe_wgrad_npu_001` | 25% | MoE sparse weight gradient 向量化点积，通过 ILP 批处理、fp32 累加与地址对齐在 910B 上提升 25%。 |

## 代表 Kernel

以下为 `exp_moe_wgrad_npu_001` 对应的 kernel：

```python
@triton.jit
def _binned_copy_wgrad_npu(
    x,
    grad,
    wgrad,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    bins_start = 0
    if expert_idx > 0:
        bins_start = tl.load(bins + expert_idx - 1)
    bins_end = tl.load(bins + expert_idx)
    expert_num_tokens = bins_end - bins_start

    for idx_in_block in range(0, BLOCK_SIZE):
        index_x = expert_idx * expert_capacity + entry_idx * BLOCK_SIZE + idx_in_block
        out_token_num = entry_idx * BLOCK_SIZE + idx_in_block + 1
        if not (out_token_num > expert_num_tokens):
            index_out = tl.load(indices + bins_start + entry_idx * BLOCK_SIZE + idx_in_block)

            wgrad_ptr = wgrad + index_out
            grad_ptr = grad + tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
            x_ptr = x + tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
            offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)

            for _ in range(iterations):
                mask = offsets < NUM_COLUMNS
                data = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
                scale = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
                acc += data * scale
                offsets += BLOCK_X

            out = tl.sum(acc).to(wgrad.dtype.element_ty)
            tl.store(wgrad_ptr, out)
```

## 已知实例

- `exp_moe_wgrad_npu_001` — Compute Optimization（收益 25%）
  - 适用于 MoE 层 sparse weight gradient 计算，关键参数 `NUM_COLUMNS ≥ 128`、`BLOCK_X ≥ 64`、`BLOCK_SIZE ≥ 1`。
  - 对应 kernel: `raw_experiences/extracted_kernels/004-_binned_copy_wgrad_npu.py`

## 关联知识

- 开发指南: [Moe 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 同属优化知识，主题相关
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](../../ops/moe-init-routing-v2-grad.md) — 主题相关：共享 moe
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](../../ops/moe-token-permute-with-ep-grad.md) — 主题相关：共享 moe
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, moe, vectorization
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute, moe
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute, vectorization
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute, vectorization
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
