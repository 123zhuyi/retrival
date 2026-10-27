---
title: Core Mapping Strategies for Matmul
category: techniques
tags: [core-mapping, matmul, diagonal, group-size, l2-reuse, pid]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Core Mapping Strategies for Matmul

## Overview

Core mapping determines which (M, N) output tile each AI Core processes. The mapping strategy directly affects L2 cache reuse, load balancing, and bank conflict patterns. Choosing the wrong strategy for a given matrix shape can leave 20-40% performance on the table.

## Three Strategies Comparison

| Strategy | Block ID Pattern | Best For | GROUP_SIZE Options | Notes |
|----------|-----------------|---------|-------------------|-------|
| Default Column | Adjacent block_ids access same N column | N-dominant (N >> M) | 8 / 16 / 32 | Simple; GROUP_2D blocks per group |
| Default Row | Adjacent block_ids access same M row | M-dominant (M >> N) | 8 / 16 / 32 | Good L2 for A matrix reuse |
| Diagonal | Adjacent block_ids access different (M,N) blocks | Square or M,N ≥ 512 | 8 / 16 / 32 | Maximizes L2 reuse, eliminates bank conflicts |

### Default Column Mapping

Adjacent block_ids are assigned to the same N column. This maximizes reuse of the B (weight) matrix along the N dimension.

```
block_ids:  0  1  2  3  4  5  6  7
M tiles:    0  1  2  3  0  1  2  3
N tiles:    0  0  0  0  1  1  1  1
```

Use when: N is the dominant dimension (tall-and-narrow weight matrices). GROUP_2D sets the number of M tiles per group.

### Default Row Mapping

Adjacent block_ids are assigned to the same M row. This maximizes reuse of the A (activation) matrix along the M dimension.

```
block_ids:  0  1  2  3  4  5  6  7
M tiles:    0  0  0  0  1  1  1  1
N tiles:    0  1  2  3  0  1  2  3
```

Use when: M is the dominant dimension (batch processing with small N).

### Diagonal Mapping

Adjacent block_ids access different (M, N) tiles arranged diagonally. This ensures that neighboring cores read from different rows and columns simultaneously, maximizing L2 reuse across both A and B matrices and eliminating L2 bank conflicts.

Use when: M ≥ 512 and N ≥ 512 (roughly square matrices, or large square-ish batched GEMM).

## GROUP_SIZE Selection Guide

GROUP_SIZE controls how many tiles form a "group" before the mapping wraps.

| GROUP_SIZE | When to Use |
|-----------|------------|
| 8 | Default; small to medium matrices |
| 16 | Medium matrices (M or N in 512–2048 range) |
| 32 | Large matrices (M or N ≥ 2048) |

Larger GROUP_SIZE increases L2 temporal reuse within a group but may reduce load balance if the grid is not evenly divisible.

## Diagonal Mapping Access Pattern Example

With GROUP_SIZE = 4 and a 4×4 tile grid:

```
Group 1 (diagonal 0): (0,0)  (1,1)  (2,2)  (3,3)
Group 2 (diagonal 1): (0,1)  (1,2)  (2,3)  (3,0)
Group 3 (diagonal 2): (0,2)  (1,3)  (2,0)  (3,1)
Group 4 (diagonal 3): (0,3)  (1,0)  (2,1)  (3,2)
```

Each group's cores read from different rows AND different columns, spreading L2 accesses evenly.

## Hardware API

**Use `dl.grouped_launch_diagonal` for hardware-level diagonal PID mapping on 910b2.**

```python
import triton_ascend.language as dl

@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    # Hardware diagonal mapping
    pid_m, pid_n = dl.grouped_launch_diagonal(
        tl.program_id(0),
        tl.cdiv(M, BLOCK_M),
        tl.cdiv(N, BLOCK_N),
        GROUP_SIZE
    )
    # pid_m, pid_n now contain the diagonally-remapped tile coordinates
```

This API performs the diagonal remapping at the hardware level, avoiding software overhead of computing the mapping manually.

**Forbidden**: Do NOT use `tl.swizzle2d` for diagonal mapping on Ascend. Use `dl.grouped_launch_diagonal` instead.

## AutoTune Parameters for Core Mapping

A minimum of 3 configs with different GROUP_SIZE values should be included in AutoTune:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 16}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 16}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 32}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 32}),
    ],
    key=['M', 'N', 'K'],
)
```

Required AutoTune parameters: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `GROUP_SIZE`.

## Decision Flowchart

```
M >= 512 AND N >= 512?
    YES → Diagonal (dl.grouped_launch_diagonal)
    NO  → N > M * 2?
              YES → Default Column
              NO  → M > N * 2?
                        YES → Default Row
                        NO  → Default Column (safe default)
```

## 相关页面
- [techniques/bank_conflict](bank_conflict.md) — How diagonal mapping eliminates L2 bank conflicts
- [techniques/tiling](tiling.md) — BLOCK_M/BLOCK_N/BLOCK_K selection
- [ops/matmul](../ops/matmul.md) — Full matmul implementation examples
- [hardware/ascend_910b](../hardware/ascend_910b.md) — Core topology and grid constraints

## 被以下页面引用
- [ops/matmul](../ops/matmul.md) — 核映射策略选择（默认列/行/对角）
- [techniques/bank_conflict](bank_conflict.md) — 对角映射消除 bank conflict
- [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) — 禁用 tl.swizzle2d 约束
- [experience_index](../experience_index.md) — skill_tiling_default_col/row/diagonal
