---
title: Tiling Strategies for Triton-Ascend
category: techniques
tags: [tiling, ub, block-size, ilp, normalization, autotune]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# Tiling Strategies for Triton-Ascend

## Overview

Tiling is the primary mechanism for mapping compute workloads onto Ascend AI Core UB memory. Incorrect tile sizing causes UB overflow, misalignment penalties, or under-utilization. This page covers the three-level hierarchy, alignment requirements, block size selection, and special patterns.

## Three-Level Hierarchy

Triton-Ascend uses a three-level tiling hierarchy:

| Level | Name | Scope | Description |
|-------|------|-------|-------------|
| L1 | `ncore` | Cross-core split | Divides total work across AI Cores (grid dimension) |
| L2 | `xblock` | Per-core data | The tile assigned to a single core's program |
| L3 | `xblock_sub` | Intra-core fine-grained | Further subdivision within a core for ILP/pipelining |

The `ncore` level determines how the grid is launched. The `xblock` level is what each `tl.program_id` processes. The `xblock_sub` level is used when a for-loop inside the kernel iterates over sub-tiles.

## UB Capacity Constraint

On Atlas 800T/I A2 hardware, each AI Core has **192KB** of Unified Buffer (UB) on-chip memory.

All tile data must fit within UB:

```
BLOCK_SIZE * element_bytes <= 192KB
```

Example capacities:
- fp16 (2 bytes): max 98,304 elements per tile
- fp32 (4 bytes): max 49,152 elements per tile
- bf16 (2 bytes): max 98,304 elements per tile

For 2D tiles (matmul):
```
BLOCK_M * BLOCK_K * dtype_bytes + BLOCK_K * BLOCK_N * dtype_bytes <= 192KB
```

## 512B Row-Width Alignment Requirement

**Mandatory**: Row widths must be aligned to 512 bytes on the Ascend backend.

- For fp16: row width must be a multiple of 256 elements (256 * 2 = 512B)
- For fp32: row width must be a multiple of 128 elements (128 * 4 = 512B)
- For int8: row width must be a multiple of 512 elements (512 * 1 = 512B)

Misalignment causes partial cache-line reads, significantly degrading memory bandwidth utilization. Always pad tensors or select BLOCK sizes that are multiples of the alignment boundary.

## BLOCK_SIZE Selection Guide

### Vectorized Operations (elementwise, normalization, reduction)

Recommended BLOCK_SIZE values: `{128, 256, 512, 1024, 2048, 4096}`

| BLOCK_SIZE | Use Case |
|-----------|---------|
| 128 | Small tensors, high-core-count parallel launch |
| 256 | Default for medium tensors |
| 512 | Normalization ops with moderate hidden dim |
| 1024 | Large hidden dims, fewer cores needed |
| 2048 | Very large hidden dims, single-pass preferred |
| 4096 | Maximum tile, single-core launch, fits 192KB with fp16 |

### Matmul Operations

Recommended BLOCK sizes: `BLOCK_M, BLOCK_N ∈ {128, 256}`, `BLOCK_K = 256`

The Cube unit has optimal efficiency with 128x128 or 256x256 tiles. Using smaller tiles results in underutilization of the Cube pipeline.

> **⚠ 矛盾说明**：此处 `BLOCK_K = 256` 特指 Cube 核心执行单元的推荐对齐粒度（Cube 单元一次处理 256 fp16 elements along K），适用于 M, N ≥ 128 的标准 GEMM。当 K 维度较小（K < 256）或在内存受限场景下，`BLOCK_K` 可取 64-128；`performance_tuning.md` 中给出的 `BLOCK_K = 64-128` 是通用建议（含小矩阵/算子融合场景）。选择原则：**K ≥ 512 时优先尝试 BLOCK_K=256；K < 256 时降至 64-128**。

## For-Loop Tiling Pattern (ILP Enablement)

Converting a single large load+compute+store into a loop of smaller tiles allows the compiler to pipeline across iterations (multi-buffer / double-buffer).

**Without for-loop (single tile, no pipeline)**:
```python
x = tl.load(ptr + offs, mask=mask)
y = compute(x)
tl.store(out_ptr + offs, y, mask=mask)
```

**With for-loop tiling (enables ILP)**:
```python
num_loop = tl.cdiv(total_elems, BLOCK_SIZE)
for i in range(num_loop):
    x = tl.load(ptr + i * BLOCK_SIZE + offs, mask=mask)
    y = compute(x)
    tl.store(out_ptr + i * BLOCK_SIZE + offs, y, mask=mask)
```

Each iteration's tile fits in UB, allowing the next tile to be prefetched while the current tile is being computed. This overlaps MTE2 (load), Vector (compute), and MTE3 (store) pipelines.

## Two-Pass Tiling for Normalization

When `n_cols > UB capacity`, normalization (LayerNorm, RMSNorm) cannot be done in a single pass. Use two-pass tiling:

**Pass 1**: Compute statistics (mean, variance, or RMS) with column blocking
```python
# Accumulate partial stats across column blocks
for col_block in range(tl.cdiv(n_cols, BLOCK_N)):
    cols = tl.load(X_ptr + col_block * BLOCK_N + offs)
    partial_sum += tl.sum(cols * cols, axis=0)
rms = tl.sqrt(partial_sum / n_cols + eps)
```

**Pass 2**: Apply normalization with the computed statistics
```python
for col_block in range(tl.cdiv(n_cols, BLOCK_N)):
    cols = tl.load(X_ptr + col_block * BLOCK_N + offs)
    normed = cols / rms * weight
    tl.store(Y_ptr + col_block * BLOCK_N + offs, normed)
```

## Column Blocking for Hidden-Dim Ops

When the hidden dimension (`n_cols`) exceeds UB capacity, process it in column blocks:

```python
BLOCK_N: tl.constexpr = 1024  # fits in UB

for col_start in range(0, n_cols, BLOCK_N):
    offs = col_start + tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    x = tl.load(X_ptr + row * n_cols + offs, mask=mask)
    # process x
    tl.store(Y_ptr + row * n_cols + offs, result, mask=mask)
```

This pattern is essential for ops on large language models where hidden dimensions are 4096, 8192, or larger.

## Layout-Driven Tile Size Selection

Tile size 的选择不仅受 UB 容量限制，还与数据布局（BTD vs BDT）耦合：

### BTD 布局 `[B, T, D]` — 适用长序列

- D 维度在最内层（stride=1），同一 token 的所有 D 维度元素连续
- **适合大 D tile**（如 BLOCK_D=1024）：一个 token 的大块 D 维度可一次 load
- 典型场景：Prefill 阶段（seq_len ≥ 512），逐 token 处理

### BDT 布局 `[B, D, T]` — 适用短序列

- T 维度在最内层（stride=1），但 seq_len 小（1 token）
- **D tile 必须小**（如 D_CHK_SIZE=16）：大的 D_CHK_SIZE 会导致跨步访问（各行间距 = seq_len），各行不在同一 cache line
- **T tile 可以大**（如 T_CHK_SIZE=256）：减少 tile 数量，降低调度开销
- 这是**异构 tile 策略**的核心原因：D 和 T 维度使用不同大小的 tile

### 异构 Tile 策略决策表

| 条件 | D tile 大小 | T tile 大小 | 原因 |
|------|------------|------------|------|
| BDT 布局, seq_len 小 | 小（16-32） | 大（128-256） | D 方向跨步访问代价高；T 方向连续，大 tile 无惩罚 |
| BTD 布局, seq_len 大 | 大（512-1024） | 小（32-64 或不做 T-tiling） | D 方向连续；大 D tile 充分利用带宽 |
| BDT 布局, seq_len 中等 | 中（32-64） | 中（64-128） | 折中：D 跨步可接受，T 无需过大 |

经验：`exp_causal_conv1d_update_007`；详见 [ops/convolution.md](../ops/convolution.md)

### 无 T-tiling 决策：寄存器滑动窗口替代方案

当序列处理使用**寄存器滑动窗口**时，T 维度无需 tile 化。窗口元素保持在寄存器中，通过移位实现跨 token 滑动，自然消除 tile 边界问题：

| 决策条件 | T-tiling（重叠区） | 无 T-tiling（寄存器滑动窗口） |
|----------|-------------------|------------------------------|
| seq_len | 任意（短序列 tile 数少时可减少调度开销） | ≥64（摊还 W 次初始化 load） |
| W（窗口大小） | 任意 | ≤8（寄存器可容纳全窗口） |
| 跨 token 依赖 | 需要 tile 边界 W-1 重叠区保证连续 | 寄存器移位自然连续，无边界 |
| 典型 kernel | Update/Decode kernel | Prefill kernel |

**决策原则**：当寄存器可容纳完整窗口（W ≤ 8）且序列足够长（seq_len ≥ 64）时，优先选择无 T-tiling + 寄存器滑动窗口——消除 T 维度 tile 管理的全部复杂度。

经验：`exp_causal_conv1d_prefill_002`、`exp_causal_conv1d_prefill_003`；详见 [ops/convolution.md](../ops/convolution.md) 和 [techniques/memory_access_optimization.md §10](memory_access_optimization.md)

### 分块递推 — 原子单元 + 层次化 Merge

对块递归算法（如分块矩阵求逆），以固定大小原子单元（如 16×16）为基本计算粒度，通过层次化 merge 支持更大的块：

```
BT=16: 原子单元直接计算（逐行递推）
BT=32: 4 块 → 对角块计算（16×16 原子）+ 1 层 merge（X₂₁ = -X₂₂·A₂₁·X₁₁）
BT=64: 16 块 → 对角块计算 + 3 层 merge（紧邻→隔行→最远），7 次 tl.dot
```

**设计原则**：
- 原子单元大小由 NPU 矩阵乘法最优粒度决定（16×16 匹配 Cube 单元）
- 所有层级的计算统一使用该原子单元（代码逻辑可复用）
- Merge 层的矩阵乘法保持 16×16 tile，充分利用 Cube 单元

**与通用 tiling 的区别**：通用 tiling 关注如何将大矩阵切分成适合 UB 的 tile；分块递推关注如何在固定原子单元上构建层次化算法。前者是硬件约束驱动，后者是算法结构驱动。

经验：`exp_solve_tril_004`；详见 [ops/solve_tril.md](../ops/solve_tril.md)

## AutoTune Configuration

Standard AutoTune config for matmul:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
    ],
    key=['M', 'N', 'K'],
)
```

Key constraint: `BLOCK_M ∈ {128, 256}`, `BLOCK_N ∈ {128, 256}`, `BLOCK_K = 256`.

## SSM / Linear Attention 反向 Tiling 要点

KDA/linear attention 反向 kernel 的 tiling 需要同时服务 Vector 标量消除和 Cube 矩阵乘利用率：

| 优化目标 | Tiling/调度选择 | 关键参数 |
|---------|----------------|---------|
| 降低 scalar | 小 BC tile + 手动指针 + fp32 mask | BC=16, BK=64/128 |
| 提升 Cube MAC | 将对角标量循环改为 `tl.dot` | 保持 BC×BC 矩阵乘 |
| 降低 MTE2 | 转置 load 改行优先 + `tl.trans` | dAqk/dAkk 加载模式 |
| 隐藏标量延迟 | 增加 num_warps | 8 warps 常见最优 |
| 软件流水线 | `tl.range` 替换 `range` | num_stages=2/3 |

**反向传播的特殊性**：
- 前向是 `chunkwise` 状态传递，反向需要同时考虑 intra-chunk（块内）和 inter-chunk（块间）梯度
- `BT` 是外层 chunk size，`BC` 是 inner tile size，通常 `BT = NC × BC`
- 对边界 chunk，需用 `T_local` 代替 `T` 计算实际行数

经验：`exp_kda_bwd_wy_scalar_opt_001`、`exp_kda_bwd_intra_scalar_opt_001`

## 相关页面
- [techniques/ilp](ilp.md) — For-loop tiling and ILP pipeline details
- [techniques/core_mapping](core_mapping.md) — Cross-core tiling (ncore level)
- [techniques/bank_conflict](bank_conflict.md) — Alignment and L2 effects
- [techniques/parallelism_and_loop](parallelism_and_loop.md) — 1D grid + 3D 任务分解（异构 tile 配合）
- [hardware/ascend_910b](../hardware/ascend_910b.md) — UB capacity and hardware constraints
- [ops/matmul](../ops/matmul.md) — Matmul-specific tiling configs
- [ops/normalization](../ops/normalization.md) — Two-pass tiling for LayerNorm/RMSNorm
- [ops/convolution](../ops/convolution.md) — BDT 布局 + 异构 tile 策略 (D_CHK_SIZE=16, T_CHK_SIZE=256)
- [ops/ssm](../ops/ssm.md) — KDA/linear attention 反向 tiling 与标量消除实例

## 被以下页面引用
- [ops/matmul](../ops/matmul.md) — matmul block size 选择
- [ops/attention](../ops/attention.md) — Q/K/V tile 大小
- [ops/normalization](../ops/normalization.md) — 列分块 two-pass（大 hidden dim）
- [ops/reduction](../ops/reduction.md) — split-K 分块
- [techniques/ilp](ilp.md) — for-loop tiling 启用 ILP
- [techniques/core_mapping](core_mapping.md) — BLOCK_M/N 与 core 映射协同
- [techniques/bank_conflict](bank_conflict.md) — 512B 对齐防 bank conflict
- [techniques/online_algorithms](online_algorithms.md) — two-pass tiling
- [techniques/operator_fusion](operator_fusion.md) — 确保融合 tile 在 UB 内
- [techniques/persistent_kernels](persistent_kernels.md) — grid-stride 的 tile 选择
- [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) — UB 约束
- [ops/convolution](../ops/convolution.md) — BDT vs BTD 布局驱动的异构 tile 策略；无 T-tiling 决策（寄存器滑动窗口替代方案）
- [ops/solve_tril](../ops/solve_tril.md) — 16×16 原子单元 + 分块递推层次化 merge 实例
