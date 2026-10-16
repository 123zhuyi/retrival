---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: activation
dimension: "2.3"
title: Activation Scheduling 优化聚合
description: 聚合 7 条 activation 算子在 Ascend 910B 上的 scheduling 优化经验，覆盖 grid-stride 循环、元素级向量化、fp32 精度保持与动态 block 调优。
tags: [activation, scheduling, grid_stride, vectorization, fp32, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/activation/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Activation Scheduling 优化聚合

## 概述

Activation 算子（SiLU、GELU、GEGLU、SwiGLU、Sigmoid 等）是 Transformer 每层都出现的基础操作。它们的计算强度很低，**性能几乎完全由内存带宽和调度方式决定**。在 Ascend 910B 上，优化重点不是 tile size，而是：
- 用 grid-stride loop 让少量 block 覆盖大行数，减少 launch 开销；
- 用向量化 load/store 提高内存总线利用率；
- 用 fp32 中间精度保证激活数值稳定；
- 对双路 gate 结构（GEGLU/SwiGLU）做融合调度。

本卡聚合 7 条 activation scheduling 经验，收益 15%–25%。

## 共同模式

### 1. Grid-Stride Loop 适配大行数

当输入行数很大而硬件 grid size 有限时，若每个 program 只处理一行或固定块，会出现 grid 不够或 launch 开销占比高的问题。Grid-stride loop 的解法是：
- grid size 固定为硬件核心数或一个较小值；
- 每个 program 通过 `for i in range(pid, total_rows, num_programs)` 处理多行；
- 减少 kernel launch 开销，同时保持合并访问。

这是 activation kernel 最常见的调度模式，7 条经验中 5 条使用。

### 2. 元素级向量化内存访问

Activation 是逐元素操作，天然适合向量化。优化要点：
- `tl.load` / `tl.store` 一次处理 BLOCK_SIZE 个元素；
- BLOCK_SIZE 对齐硬件 burst length（如 128 字节）；
- 用 `tl.where` 替代分支，避免 SIMD divergence。

### 3. fp32 中间精度保持

GELU、SiLU、SwiGLU 等激活涉及 exp、erf、sigmoid 等非线性函数，对输入范围敏感。在 fp16/bf16 输入上直接计算容易溢出或精度损失。标准做法：
- load 后先 cast 到 fp32；
- 计算激活；
- store 前再 cast 回输出精度。

收益不是最高的，但是保证训练/推理精度的必要基线。

### 4. 双路 Gate 融合调度（GEGLU / SwiGLU）

GEGLU 和 SwiGLU 需要同时加载 gate 和 value 两个张量，进行元素乘。优化关键是：
- 把两个 load 合并到同一个 program，避免两次遍历；
- 在寄存器内完成 gate 计算和逐元素乘；
- 只产生一次 store。

对于内存带宽受限的 MLP 层，这能显著降低 HBM 压力。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_silu_bwd_001` | 15% | SiLU backward：autotune、grid-stride、fp32 safeguard、向量化 |
| `exp_silu_fwd_001` | 25% | SiLU forward：grid-stride、动态 block、fp32、合并访问 |
| `exp_geglu_flat_001` | 25% | GEGLU flat：grid-stride、内存 flatten、tanh 近似 GELU、软件流水 |
| `exp_gelu_fwd_001` | 15% | GELU forward：grid-stride、autotune BLOCK_M、tanh 近似 |
| `exp_gelu_bwd_001` | 20% | GELU backward：autotune、grid-stride、向量化、精度 upcast |
| `exp_sigmoid_grid_stride_001` | 15% | Sigmoid：grid-stride、精度稳定、向量化 block |
| `exp_swiglu_fwd_001` | 15% | SwiGLU forward：混合精度、向量化、gate 融合、grid-stride |

## 代表 Kernel

以下为 `exp_swiglu_fwd_001` 对应的 kernel，收益 15%，体现了 **双路 gate 融合 + 向量化 + grid-stride + fp32 中间精度**：

```python
@triton.jit
def _swiglu_fwd_kernel(
    x_ptr, y_ptr, out_ptr,
    n_rows, n_cols,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    num_pid_m = tl.num_programs(0)

    for row_idx in range(pid_m, n_rows, num_pid_m):
        col_offsets = tl.arange(0, BLOCK_N)
        mask = col_offsets < n_cols

        # gate 和 value 通常是同一个输入的两部分，或两个输入
        x_row = x_ptr + row_idx * stride_xm
        y_row = y_ptr + row_idx * stride_ym
        out_row = out_ptr + row_idx * stride_outm

        x = tl.load(x_row + col_offsets * stride_xn, mask=mask).to(tl.float32)
        y = tl.load(y_row + col_offsets * stride_yn, mask=mask).to(tl.float32)

        # SwiGLU: x * sigmoid(x) * y
        gate = x * tl.sigmoid(x)
        out = gate * y

        tl.store(out_row + col_offsets * stride_outn,
                 out.to(out_ptr.dtype.element_ty), mask=mask)
```

关键实现细节：
- grid-stride 外循环让一个 program 处理多行；
- gate 和 value 在同一 kernel 内加载和融合；
- 中间计算在 fp32 完成，避免 fp16 sigmoid 精度问题；
- 输出前 downcast，减少 HBM 写带宽。

## 已知实例

- `exp_silu_bwd_001` — Scheduling（收益 15%）
  - SiLU backward：保守 autotune（最大 block 32）、grid-stride、fp32 safeguard、向量化内存访问。
- `exp_silu_fwd_001` — Scheduling（收益 25%）
  - SiLU forward：grid-stride 循环覆盖大行数，动态 block 调优，fp32 精度保持，合并访问。
- `exp_geglu_flat_001` — Scheduling（收益 25%）
  - GEGLU flat：grid-stride、内存 flatten、tanh 近似 GELU、软件流水。
- `exp_gelu_fwd_001` — Scheduling（收益 15%）
  - GELU forward：grid-stride、autotune BLOCK_M、tanh 近似，适合大 batch。
- `exp_gelu_bwd_001` — Scheduling（收益 20%）
  - GELU backward：autotune、grid-stride、向量化内存访问、精度 upcast。
- `exp_sigmoid_grid_stride_001` — Scheduling（收益 15%）
  - Sigmoid：grid-stride 调度，精度稳定，向量化 block 访问。
- `exp_swiglu_fwd_001` — Scheduling（收益 15%）
  - SwiGLU forward：混合精度、向量化、gate 融合、grid-stride，代表 kernel 见上。

## 关联知识

- 开发指南: [Elementwise / Activation 算子优化](../../reference/op-class-elementwise.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 主题相关：共享 fp32
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 fp32
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 fp32
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 主题相关：共享 fp32
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 fp32
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 主题相关：共享 fp32
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 主题相关：共享 fp32
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 910b2, 910b3, fp32
- 相关主题: [Activation Backward 内存访问优化聚合](../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, activation
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 grid_stride
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, activation
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
