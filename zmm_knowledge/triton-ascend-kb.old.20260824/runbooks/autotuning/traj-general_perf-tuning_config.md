---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_perf
dimension: "5.1"
title: 扩展 Autotune 配置空间以覆盖多 Shape 最优调度
description: 针对 Ascend 910B 上单一 autotune 配置无法覆盖不同 shape 和寄存器压力的问题，说明如何扩展 num_warps/num_stages/tile 组合，典型收益 10–20%。
tags: [general_perf, tuning_config, autotune, num_warps, num_stages, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_perf/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 扩展 Autotune 配置空间以覆盖多 Shape 最优调度

## 概述

Triton 的 `@triton.autotune` 可以在编译期自动搜索一组配置并选择最快的一个。但很多开发者只配置单一组合（如 `num_warps=2, num_stages=3`），这在不同输入 shape、不同寄存器压力下往往不是最优。该经验建议 **系统性地扩展 autotune 配置空间**，覆盖更多 `num_warps`、`num_stages` 和 tile 大小组合。

核心结论：

- 单一配置通常无法同时覆盖小 shape、大 shape、高寄存器压力、低寄存器压力等多种场景；
- 对 Ascend 910B，`num_warps=8` 通常能更好隐藏标量延迟，但高寄存器压力 kernel 可能 spill；
- `num_stages` 影响流水线深度，需与内存访问模式匹配；
- tile 大小（BK/BV/BLOCK_M/BLOCK_N 等）直接决定 Cube 利用率和寄存器占用。

典型收益：从单一配置到完整 tuned 配置可再加速 10–20%（如 `chunk_kda_bwd_kernel_wy_dqkg_fused` 上 BK=128、BV=64、num_warps=8、num_stages=2 最优）。

## 共同模式

### 1. num_warps 覆盖 2/4/8

- `num_warps=2`：适合寄存器压力大、live variable 多的 kernel；
- `num_warps=4`：平衡选择；
- `num_warps=8`：适合需要隐藏标量延迟、寄存器压力较小的 kernel，在 Ascend 上常能取得最佳。

### 2. num_stages 覆盖 2/3/4

- `num_stages=2`：保守选择，适合内存访问延迟不突出的 kernel；
- `num_stages=3/4`：增加流水线深度，可以更好隐藏 HBM 延迟，但会占用更多寄存器。

### 3. tile 大小多维覆盖

除了 warp/stage，还应覆盖不同 BK/BV/BLOCK_M/BLOCK_N 组合。例如：

```python
triton.Config({"BK": 64,  "BV": 64},  num_warps=4, num_stages=2),
triton.Config({"BK": 128, "BV": 64},  num_warps=8, num_stages=2),
triton.Config({"BK": 128, "BV": 128}, num_warps=8, num_stages=3),
```

### 4. 开发阶段 vs 最终版本

- **开发阶段**：使用小配置集（2–4 个）快速迭代，减少编译时间；
- **最终版本**：启用完整搜索，虽然首次编译时间增加，但对重复运行收益显著。

### 5. 验证寄存器 spill

高 warp 数会提高每个 program 的寄存器需求。如果 kernel 已经使用了大量累加器（如多个 `BC×BK` 块），增加 warp 数可能导致寄存器 spill 到 HBM，反而降低性能。应通过 profiling 确认实际寄存器使用情况。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_perf_017` | 20% | 扩展 autotune 配置空间（num_warps/num_stages/tile 组合），覆盖不同 shape 和寄存器压力，典型收益 10–20%。 |

## 代表 Kernel

`exp_perf_017` 没有独立的 extracted kernel 文件，以下为 YAML 中的配置示例：

```python
# 单一配置（空间窄）
@triton.autotune(
    configs=[triton.Config({}, num_warps=2, num_stages=3)],
    key=["BK", "NC", "BT"],
)

# 扩展配置空间
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["BK", "NC", "BT"],
)
```

## 已知实例

- `exp_perf_017` — Tuning Config（收益 20%）
  - 适用于需要 `@triton.autotune` 的通用 Triton kernel；在 `chunk_kda_bwd_kernel_wy_dqkg_fused` 上最优为 BK=128、BV=64、num_warps=8、num_stages=2。
  - 来源: `fla/ops/kda/zy_design.md`、`fla/ops/kda/intra_design.md`。

## 关联知识

- 开发指南: [General_Perf 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 实践案例: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 主题相关：共享 910b2, 910b3, autotune
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](../compute-pipeline/for-loop-tiling-ilp.md) — 主题相关：共享 num_stages
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3, autotune
- 相关主题: [Ascend 大 Shape 算子的交错循环负载均衡](../compute-pipeline/traj-general_ascend-load_balance.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [复杂算子拆分：降低编译与调度开销](../compute-pipeline/traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Element-wise 算子避免冗余 Reshape](../compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化](../compute-pipeline/traj-general_perf-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 autotune
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [输出张量使用 empty_like 替代 zeros_like](../memory-sram/traj-general_perf-memory_init.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](autotune-debug.md) — 主题相关：共享 autotune
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) — 主题相关：共享 autotune, num_stages, num_warps
- 相关主题: [移除冗余 compile_hint 恢复编译器优化](traj-general_ascend-tuning_config.md) — 主题相关：共享 910b2, 910b3, tuning_config

<!-- okf:related:end -->
