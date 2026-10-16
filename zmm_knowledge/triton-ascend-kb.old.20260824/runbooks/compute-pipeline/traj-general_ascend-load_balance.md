---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_ascend
dimension: "2.2"
title: Ascend 大 Shape 算子的交错循环负载均衡
description: 针对 Ascend 910B 大 shape 算子，通过固定 grid 为核心数、kernel 内以步长交错循环的方式分配任务，实现各核心负载差不超过 1，避免 grid 超限并提升 10–15%。
tags: [general_ascend, load_balance, stride_loop, grid_limit, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_ascend/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Ascend 大 Shape 算子的交错循环负载均衡

## 概述

在 Ascend 910B 上，大 shape 算子（元素数 ≥ 1e6 或行数很大）如果把 grid 直接设为数据块数量，很容易遇到两类问题：

1. **Grid 超限**：1D grid 在某些平台有 65535 的上限，2D grid 乘积也可能超限；
2. **负载不均**：按连续块划分时，尾部 block 可能数据不足，导致部分核心空闲。

该组经验采用 **固定 grid 为硬件核心数、kernel 内以 `core_num` 为步长交错循环** 的调度模式：每个 program 只处理 `pid, pid+CORE_NUM, pid+2*CORE_NUM, ...` 对应的任务。这样：

- 各核心处理的任务数差不超过 1，天然负载均衡；
- grid 大小固定为 VEC/CUBE 核心数，彻底规避 grid 超限；
- 代码简洁，无需计算 `ELEMENTS_PER_GRID` 等复杂参数；
- 适配按行/按块独立处理的场景（element-wise、逐行 softmax、归约、matmul 等）。

收益：`exp_ascend_grid_large_shape_001` 提升 15%；`exp_ascend_parallel_pattern_001` 提升 10% 且代码量减少 30%。

## 共同模式

### 1. 固定 grid 为物理核心数

host 侧通过 `torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)` 获取当前设备的 VEC 核心数（910B2 默认 48，910B3 默认 64），并以此作为 grid 大小。CUBE 场景类似获取 cube core 数。

### 2. Kernel 内交错循环

每个 program 的 `pid` 固定，主循环写成 `for row_idx in range(pid, M, CORE_NUM)`。这样第 `pid` 个核心依次处理第 `pid`、`pid+CORE_NUM`、`pid+2*CORE_NUM` ... 行，所有行被恰好覆盖一次。

### 3. 列方向仍做 block 分块

在行维度交错的同时，列维度仍按 `BLOCK_N` 分块处理，保证向量宽度和缓存效率。例如：

```python
for col_start in range(0, N, BLOCK_N):
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
```

### 4. VEC vs CUBE 场景适配

- **VEC 核心**：适合逐行独立处理的算子，如 element-wise、逐行 softmax/归约；
- **CUBE 核心**：适合矩阵乘法类算子，如 matmul、attention 中的 `tl.dot`。

两种场景都使用相同的 `range(pid, total_items, core_num)` 分配模式。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_ascend_grid_large_shape_001` | 15% | 大 shape 算子固定 grid 为核心数，kernel 内交错循环处理，负载均衡且规避 grid 超限。 |
| `exp_ascend_parallel_pattern_001` | 10% | `for i in range(pid, total_items, core_num)` 是 Ascend 最优并行任务分配模式，代码量减少 30%。 |

## 代表 Kernel

以下为 `exp_ascend_parallel_pattern_001` 对应的 inline 代码示例：

```python
@triton.jit
def parallel_assign_kernel(
    input_ptr, output_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    # 经典并行分配模式：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM... 行
    for row_idx in range(pid, M, CORE_NUM):
        row_ptr = input_ptr + row_idx * stride_m
        out_row_ptr = output_ptr + row_idx * stride_m

        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            result = tl.nn.softmax(data, axis=0)  # 逐行 softmax（VEC 核心适配场景）
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)
```

## 已知实例

- `exp_ascend_grid_large_shape_001` — 负载均衡优化（收益 15%）
  - 适用于元素数 ≥ 1e6 的大 shape 算子，2D/3D 张量，按行/按块独立处理。
- `exp_ascend_parallel_pattern_001` — 负载均衡优化（收益 10%）
  - 适用于总任务数远大于核心数的大 shape 场景；VEC 核心用于逐行处理，CUBE 核心用于矩阵乘类算子。

## 关联知识

<!-- okf:related:start -->

# 相关

- 相关主题: [移除冗余 compile_hint 恢复编译器优化](../autotuning/traj-general_ascend-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 输出张量创建优化：empty/empty_like 替代 zeros/ones](../memory-sram/traj-general_ascend-memory_init.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端避免用 tl.where 动态计算内存偏移](../meta-skills/traj-general_ascend-syntax_adapt.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](../meta-skills/traj-general_ascend-syntax_pitfall.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 硬件资源适配规则聚合](../portability/traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 load_balance
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 主题相关：共享 grid_limit
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 主题相关：共享 load_balance

<!-- okf:related:end -->
