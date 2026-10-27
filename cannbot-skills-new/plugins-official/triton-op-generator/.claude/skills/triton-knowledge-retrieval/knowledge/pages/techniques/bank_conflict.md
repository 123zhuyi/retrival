---
title: L2 Bank Conflicts and Mitigation on Ascend
category: techniques
tags: [bank-conflict, l2-cache, diagonal-tiling, memory, aiv-mte2]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# L2 Bank Conflicts and Mitigation on Ascend

## Overview

L2 bank conflicts are a primary cause of memory bottlenecks in large matmul kernels on Ascend. Understanding the cause and applying diagonal core mapping is essential for achieving peak throughput on square or large matrices.

## What Is an L2 Bank Conflict?

The Ascend L2 cache is divided into multiple banks. When multiple AI Cores simultaneously request data that maps to the same L2 bank, the accesses are serialized — only one can be served per cycle. This is called a **bank conflict**.

**Root cause in matmul**: With default column or row mapping, adjacent block_ids access the same N column (or M row). This means:
- Multiple cores load the same B matrix tile simultaneously → same L2 bank
- Multiple cores load adjacent A matrix rows → adjacent L2 banks (less conflict, but still problematic at scale)

**Effect**: Effective memory bandwidth drops significantly below the hardware peak of ~1.8 TB/s. The serialization shows up as elevated `aiv_mte2_time` in profiling.

## How Diagonal Tiling Eliminates Bank Conflicts

With diagonal mapping, adjacent block_ids are assigned to tiles at different (M, N) positions:

**Default Column (conflicts)**:
```
Core 0: A[row=0..127, :] × B[:, col=0..127]   ← same B columns as Core 1
Core 1: A[row=128..255, :] × B[:, col=0..127]  ← same B columns as Core 0
Core 2: A[row=256..383, :] × B[:, col=0..127]  ← same B columns as Core 0,1
```
Cores 0, 1, 2 all load the same B tile → bank conflict.

**Diagonal Mapping (no conflicts)**:
```
Core 0: A[row=0..127, :] × B[:, col=0..127]    ← unique B column
Core 1: A[row=128..255, :] × B[:, col=128..255] ← unique B column
Core 2: A[row=256..383, :] × B[:, col=256..383] ← unique B column
```
Each core reads from different L2 banks → no serialization.

## Key Profiling Metric

**`aiv_mte2_time`**: Time spent on MTE2 (Memory Transfer Engine for loads from HBM/L2 to UB).

| aiv_mte2_time | Interpretation |
|--------------|----------------|
| Low (close to theoretical) | Memory access is efficient; no significant bank conflict |
| High (2-5x theoretical) | Memory bottleneck; likely bank conflict or misalignment |
| Very high (>5x theoretical) | Severe bank conflict or completely unaligned access |

To compute theoretical MTE2 time:
```
Theoretical_time = Data_volume / HBM_bandwidth
                 = (M * K + K * N + M * N) * dtype_bytes / 1.8e12
```

If measured `aiv_mte2_time` >> theoretical, bank conflict is the primary bottleneck.

## Solution: Diagonal Core Mapping

Apply `dl.grouped_launch_diagonal` to remap PIDs to diagonal positions:

```python
import triton_ascend.language as dl

@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Diagonal remapping: adjacent PIDs → different (M, N) positions
    pid_m, pid_n = dl.grouped_launch_diagonal(pid, num_pid_m, num_pid_n, GROUP_SIZE)

    # Rest of kernel unchanged
    ...
```

This single change is typically sufficient to eliminate L2 bank conflicts for large square matmul.

## Also: 512B Alignment Helps Avoid Partial Cache-Line Reads

Unaligned memory access causes partial L2 cache-line reads. When a load spans two cache lines:
1. Two L2 reads are issued instead of one
2. Effective bandwidth is halved
3. Probability of bank conflict doubles

**Rule**: Ensure all tensor row widths are multiples of 512 bytes. See [techniques/tiling](tiling.md) for alignment requirements.

## PID Grouping for L2 Locality: Adjacent PIDs Should Share K Slices

Beyond diagonal mapping, PID grouping within a group should ensure that adjacent PIDs within a group share K-dimension data in L2:

- Cores in the same group process the same K range
- The K slices loaded by one core can be reused by adjacent cores via L2
- This applies to both split-K and normal GEMM

GROUP_SIZE controls the number of tiles in a group. Larger GROUP_SIZE increases K-dimension L2 reuse but may reduce spatial diversity within the group.

**Optimal**: GROUP_SIZE = 8 (default). For very large K, consider GROUP_SIZE = 16 or 32.

## Diagnostic Checklist

1. Run profiling to obtain `aiv_mte2_time`
2. Compute theoretical MTE2 time
3. If ratio > 2x: suspect bank conflict
4. Check matrix shape: M ≥ 512 AND N ≥ 512 → apply diagonal mapping
5. Check alignment: row width % 512B == 0
6. Re-profile to confirm improvement

## 相关页面
- [techniques/core_mapping](core_mapping.md) — Diagonal mapping implementation and GROUP_SIZE selection
- [techniques/tiling](tiling.md) — 512B alignment requirement
- [workflows/profiling](../workflows/profiling.md) — How to measure aiv_mte2_time
- [ops/matmul](../ops/matmul.md) — Full matmul kernel with diagonal mapping

## 被以下页面引用
- [ops/matmul](../ops/matmul.md) — 大矩阵对角 tiling 消除 bank conflict
- [ops/attention](../ops/attention.md) — attention score 矩阵访问模式
- [techniques/core_mapping](core_mapping.md) — 对角映射实现细节
- [workflows/profiling](../workflows/profiling.md) — aiv_mte2_time 诊断
