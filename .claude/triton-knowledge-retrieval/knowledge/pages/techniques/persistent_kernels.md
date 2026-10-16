---
title: Persistent Kernel Pattern
category: techniques
tags: [persistent-kernel, grid-stride, async-task, cube-vector, throughput, inference]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Persistent Kernel Pattern

## Overview

A persistent kernel fixes the grid size to the number of AI Cores (`num_aicores`) and uses an internal grid-stride loop to process multiple work units per core. This avoids repeated kernel launch overhead and enables weight/constant data to be loaded once and reused for many output rows.

## Concept

**Regular kernel**: Grid = number of work units. Each core processes exactly one unit, then terminates. For large workloads, many kernel launches are needed, each incurring startup cost. Weight vectors loaded separately per launch.

**Persistent kernel**: Grid = `num_aicores` (fixed). Each core runs a loop over all work units assigned to it. Weight vectors loaded once at the start of the kernel, reused for all iterations.

```
Regular:    [launch][core0: row0] [launch][core0: row1] [launch][core0: row2] ...
Persistent: [launch] [core0: row0 → row4 → row8 → ...] (single launch, loop)
```

## Benefit

**25-35% throughput boost** when:
- `n_rows >> num_aicores` (more rows than cores)
- Weight/constant tensors are the same for all rows (bias, norm weights, embedding tables)
- Per-row setup cost (loading weights) is significant relative to per-row compute cost

This pattern is especially effective during inference with large batch sizes, where the same model weights are applied to many input rows.

## When to Use

| Condition | Use Persistent? |
|-----------|----------------|
| n_rows >> num_aicores | Yes |
| n_rows ≈ num_aicores | No benefit |
| Weight loaded per row | Yes (amortize load) |
| Stateless per-row ops | Optional |
| Dynamic row count | Yes (handles variable batch at fixed grid) |

## Grid-Stride Pattern

```python
import triton
import triton.language as tl
import triton_ascend.language as dl

@triton.jit
def persistent_layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Fixed grid: num_aicores programs total
    num_programs = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    # Load weights once per program (amortized over all rows)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    weight = tl.load(W_ptr + offs, mask=mask)
    bias = tl.load(B_ptr + offs, mask=mask)

    # Grid-stride loop: each core handles rows [pid, pid+num_programs, pid+2*num_programs, ...]
    for row in range(pid, n_rows, num_programs):
        # Load row
        x = tl.load(X_ptr + row * n_cols + offs, mask=mask)

        # Normalize
        mean = tl.sum(x, axis=0) / n_cols
        var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
        x_norm = (x - mean) / tl.sqrt(var + eps)

        # Apply affine transform
        y = x_norm * weight + bias
        tl.store(Y_ptr + row * n_cols + offs, y, mask=mask)
```

Launch:
```python
# Fixed grid = number of AI Cores
grid = (num_aicores,)
persistent_layernorm_kernel[grid](X, W, B, Y, n_rows, n_cols, eps, BLOCK_SIZE=1024)
```

## `dl.async_task`: Cube/Vector Async Heterogeneous Parallelism

For fused kernels that mix GEMM (Cube) and elementwise (Vector) ops, `dl.async_task` enables the two units to run concurrently.

```python
@triton.jit
def fused_persistent_gemm_norm_kernel(...):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for row_block in range(pid, n_row_blocks, num_programs):
        # Cube unit: GEMM
        with dl.async_task(scope="cube"):
            acc = tl.dot(a_tile, b_tile)

        # Vector unit: normalization (runs while Cube loads next tile)
        with dl.async_task(scope="vector"):
            rms = tl.sqrt(tl.sum(acc * acc, axis=1) / n_cols + eps)
            result = acc / rms[:, None] * weight
            tl.store(Y_ptr + ..., result)
```

### Scope Values

| scope | Unit | Typical Operations |
|-------|------|-------------------|
| `"cube"` | Cube unit | tl.dot, GEMM, batch matmul |
| `"vector"` | Vector unit | elementwise, tl.exp, normalization, reduction |

### Pipeline Overlap

With `dl.async_task`:
```
Iteration i:   [Cube: GEMM tile i] ──────────────────────→
                                   [Vector: activate tile i-1] →
Iteration i+1: [Cube: GEMM tile i+1] ─────────────────────→
                                      [Vector: activate tile i] →
```

The Vector unit processes tile `i-1` while the Cube unit computes tile `i`.

## Persistent Kernel vs Regular Kernel: When Each Wins

| Scenario | Better Choice | Reason |
|---------|--------------|--------|
| n_rows >> num_aicores | Persistent | Amortize launch + weight load |
| n_rows ≈ num_aicores | Regular | Grid exactly matches work, no loop overhead |
| n_rows < num_aicores | Regular (pad) | Persistent wastes idle cores |
| Weight tensor is large | Persistent | Load once, reuse for all rows |
| Per-row work is very small | Regular | Loop overhead dominates in persistent |
| Fused GEMM + activation | Persistent + async_task | Cube/Vector overlap |
| One-shot large matmul | Regular | Single tile per core, no weight sharing |

**Rule of thumb**: Use persistent kernel when `startup cost > per-iteration cost`. For LayerNorm on n_rows=8192 with num_aicores=20, the persistent kernel processes 410 rows per core — far amortizing the launch overhead.

## 相关页面
- [techniques/ilp](ilp.md) — dl.async_task and pipeline overlap details
- [techniques/tiling](tiling.md) — BLOCK_SIZE selection for grid-stride loops
- [hardware/ascend_910b](../hardware/ascend_910b.md) — num_aicores detection
- [ops/normalization](../ops/normalization.md) — Persistent LayerNorm/RMSNorm examples
- [techniques/operator_fusion](operator_fusion.md) — Combining persistent pattern with fusion

## 被以下页面引用
- [ops/normalization](../ops/normalization.md) — RMSNorm/LayerNorm persistent kernel（kernel 086）
- [ops/matmul](../ops/matmul.md) — matmul persistent kernel（exp_matmul_sched_003）
- [ops/reduction](../ops/reduction.md) — reduction persistent pattern
- [techniques/operator_fusion](operator_fusion.md) — persistent + fusion 组合
