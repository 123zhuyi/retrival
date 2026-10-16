---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: activation
dimension: "1.1"
title: Activation Backward 内存访问优化聚合
description: 聚合 2 条 activation backward（GEGLU、SwiGLU flat）在 Ascend 910B 上的 memory access 经验，覆盖 flatten 内存布局、激活重计算、向量化 load/store 与 grid-stride 循环。
tags: [activation, memory_access, backward, geglu, swiglu, activation_recomputation, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/activation/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Activation Backward 内存访问优化聚合

## 概述

GEGLU 和 SwiGLU 是现代 LLM MLP 层常用的激活结构，其 backward 需要同时访问 gate 和 value 两个分支的梯度。由于 MLP 中间维度通常很大，**内存带宽是 backward 的主要瓶颈**。本卡聚合 2 条 activation backward 的 memory access 经验，核心思路是：把多维张量 flatten 成一维连续布局，用 grid-stride 循环覆盖大行数，通过激活重计算避免保存中间激活，并用向量化 load/store 提高带宽利用率。

## 共同模式

### 1. Flatten 内存布局

GEGLU/SwiGLU 的输入通常是 2D/3D（如 `[batch*seq, 2*hidden]` 或 `[batch, seq, 2*hidden]`）。Backward 时若按多维 stride 寻址，会破坏合并访问。优化做法：
- 在 kernel 调用前把张量 reshape/flatten 为 1D 或 2D 连续布局；
- kernel 内部用简单的线性地址计算；
- 保证相邻线程访问相邻内存。

### 2. 激活重计算（Activation Recomputation

GEGLU backward 需要 gate 和 value 的前向值。如果把这些中间激活保存下来，会占用大量 HBM。优化做法：
- 在 backward kernel 中根据输入重新计算 gate/value；
- 用重计算的额外 Vector 计算换取 HBM 带宽和容量；
- 对于 bandwidth-bound 的 MLP backward，重计算通常是正收益。

### 3. Grid-Stride + 向量化 Load/Store

Activation backward 是元素级操作，适合：
- grid-stride 循环让一个 program 处理多个元素块；
- `tl.load`/`tl.store` 一次性处理 BLOCK_SIZE 个元素；
- BLOCK_SIZE 对齐硬件事务宽度（如 128 字节）。

### 4. 精度保护

Backward 中的梯度通常需要在 fp32 上计算，尤其是涉及 sigmoid/gelu 导数的部分。加载后先 upcast 到 fp32，计算完成后再 downcast 回梯度精度。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_geglu_bwd_001` | 25% | GEGLU backward flat：grid-stride、激活重计算、软件流水、精度保持 |
| `exp_swiglu_flat_001` | 25% | SwiGLU forward flat：grid-stride、软件流水、向量化、精度安全 |

## 代表 Kernel

以下为 `exp_geglu_bwd_001` 对应的 kernel，体现了 **flatten 布局 + 激活重计算 + grid-stride + 向量化**：

```python
@triton.jit
def _geglu_backward_kernel_flat(
    grad_out_ptr, input_ptr, grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    for off in range(pid * BLOCK_SIZE, n_elements, num_pid * BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # 加载 flat 布局的输入和梯度
        x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask).to(tl.float32)

        # 重计算 gate 和 value
        # 假设输入沿最后一维拼接了 gate 和 value
        mid = n_elements // 2
        is_gate = offsets < mid

        # GEGLU: gate = x * GELU(x), output = gate * value
        # backward 需要根据具体结构计算
        gate = tl.where(is_gate, x, 0.0)
        value = tl.where(is_gate, 0.0, x)
        gelu_gate = gelu(gate)
        grad_gelu = grad_out * value * gelu_derivative(gate)
        grad_value = grad_out * gelu_gate

        grad_input = tl.where(is_gate, grad_gelu, grad_value)
        tl.store(grad_input_ptr + offsets, grad_input.to(grad_input_ptr.dtype.element_ty), mask=mask)
```

关键实现细节：
- 输入按 flat 1D 布局处理；
- gate/value 在 backward 时重计算，避免保存中间激活；
- grid-stride 循环覆盖所有元素；
- 中间计算在 fp32 完成。

## 已知实例

- `exp_geglu_bwd_001` — Memory Access（收益 25%）
  - GEGLU backward flat：flatten 1D 布局简化寻址，激活重计算减少 HBM 占用，grid-stride + 软件流水 + 精度保持。大 batch/seq 场景收益明显。
- `exp_swiglu_flat_001` — Memory Access（收益 25%）
  - SwiGLU forward flat：grid-stride、软件流水、向量化内存访问、精度安全。要求输入 contiguous 或预先 flatten，BLOCK_SIZE 对齐 cache line。

## 关联知识

- 开发指南: [Elementwise / Activation 算子优化](../../reference/op-class-elementwise.md)
- 相关 runbook: [Activation Scheduling 优化聚合](../compute-pipeline/traj-activation-scheduling.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 activation
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](../compute-pipeline/traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, activation, swiglu
- 相关主题: [Activation Scheduling 优化聚合](../compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, activation
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, backward, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
