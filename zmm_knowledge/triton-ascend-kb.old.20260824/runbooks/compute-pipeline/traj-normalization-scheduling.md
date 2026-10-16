---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: normalization
dimension: "2.3"
title: Normalization Scheduling 优化聚合
description: 聚合 2 条 normalization 在 Ascend 910B 上的 scheduling 经验：grid-stride 循环适配变长 batch，以及 two-pass 算法平衡寄存器压力与内存带宽。
tags: [normalization, scheduling, grid_stride, two_pass, rmsnorm, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/normalization/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Normalization Scheduling 优化聚合

## 概述

Normalization 算子虽然计算简单，但**调度方式直接影响其在变长 batch、KV-LoRA 等特殊结构上的效率**。本卡聚合 2 条 scheduling 经验，覆盖两个核心问题：
- 如何用 grid-stride 循环让少量 program 覆盖大量 row，避免重新编译；
- 如何在 RMSNorm 等需要统计量的算子中，用 two-pass 算法平衡寄存器压力和内存带宽。

虽然只有 2 条 experience，但它们代表的模式在几乎所有 normalization kernel 中都会遇到。

## 共同模式

### 1. Grid-Stride 循环适配变长 Batch

RMSNorm/LayerNorm 通常按 row 独立计算。当 batch 大小（row 数）变化时，如果 grid size 与 row 数绑定，每次不同 batch 都会触发重新编译。grid-stride 的解法是：
- grid size 固定为硬件核心数；
- 每个 program 通过 `for r in range(pid, n_rows, num_programs)` 处理多行；
- n_rows 作为运行时参数传入，无需重新编译。

收益：
- 减少 JIT 编译开销；
- 提高核心利用率，小 batch 也能占满硬件；
- 代码更简洁，无需为不同 batch size 写特殊逻辑。

### 2. Two-Pass 算法平衡寄存器与带宽

RMSNorm 需要计算 `sum(x²)` 和 `x * rstd * w`。如果一次性把整行加载到寄存器再计算：
- 大 hidden dim 时寄存器压力高，容易 spill；
- 但省下一次遍历的 HBM 带宽。

Two-pass 做法是：
- 第一遍：分块加载，累加 `sum(x²)`，得到 rstd；
- 第二遍：再次分块加载 x，应用 `rstd * w`，store 结果。

收益与权衡：
- 多一次 x 的读取；
- 但寄存器占用显著降低，可以支持更大的 hidden dim；
- 对 HBM 带宽充足的场景通常是更好的选择。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_rmsnorm_fwd_001` | 25% | RMSNorm forward：grid-stride 适配变长 n_rows，two-pass 算法 |
| `exp_rmsnorm_kv_001` | 25% | RMSNorm on KV LoRA rank：grid-stride、weight 向量缓存、混合精度累加 |

## 代表 Kernel

以下为 `exp_rmsnorm_fwd_001` 对应的 kernel，体现了 **grid-stride + two-pass**：

```python
@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr,
    n_rows, n_cols, eps: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    # grid-stride 外循环：一个 program 处理多行
    for row_idx in range(pid, n_rows, num_pid):
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = col_offsets < n_cols

        # Pass 1: 计算 sum(x^2)
        sum_sq = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_COLS):
            cols = col_start + col_offsets
            m = cols < n_cols
            x = tl.load(x_ptr + row_idx * n_cols + cols, mask=m, other=0.0).to(tl.float32)
            sum_sq += x * x
        variance = tl.sum(sum_sq, axis=0) / n_cols
        rstd = tl.rsqrt(variance + eps)

        # Pass 2: 应用 rstd 和 weight
        w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        for col_start in range(0, n_cols, BLOCK_COLS):
            cols = col_start + col_offsets
            m = cols < n_cols
            x = tl.load(x_ptr + row_idx * n_cols + cols, mask=m, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(y_ptr + row_idx * n_cols + cols,
                     y.to(y_ptr.dtype.element_ty), mask=m)
```

关键实现细节：
- `for row_idx in range(pid, n_rows, num_pid)` 是标准 grid-stride；
- `n_rows` 运行时传入，无需 constexpr；
- two-pass 把统计量计算和最终变换分开，降低寄存器压力；
- 所有累加在 fp32 完成。

## 已知实例

- `exp_rmsnorm_fwd_001` — Scheduling（收益 25%）
  - RMSNorm forward：grid-stride 循环适配变长 batch，two-pass 算法平衡寄存器与带宽，自适应精度累加。
- `exp_rmsnorm_kv_001` — Scheduling（收益 25%）
  - 对 KV LoRA rank 维度做 RMSNorm：grid-stride、weight 向量缓存、混合精度累加、strided 访问。

## 关联知识

- 开发指南: [Normalization 算子优化](../../reference/op-class-normalization.md)
- 相关 runbook: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, normalization, rmsnorm
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 rmsnorm
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rmsnorm
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 rmsnorm
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 normalization, rmsnorm
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, normalization, rmsnorm
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, normalization, rmsnorm, two_pass
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 rmsnorm
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 grid_stride
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
