---
title: Ascend 910B2/910B3 Hardware Reference
category: hardware
tags: [910b2, 910b3, ub, hbm, cube, vector, grid, mte, compilation]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Ascend 910B2/910B3 Hardware Reference

## Core Architecture

Each Ascend AI Core consists of two distinct compute units:

| Unit | Purpose | Operations |
|------|---------|-----------|
| **Cube unit** | GEMM / matrix multiply | tl.dot, batch matmul, convolution |
| **Vector unit** | Elementwise / normalization | tl.exp, tl.sum, arithmetic, LayerNorm |

These two units can execute concurrently within a single AI Core when scheduled correctly (see `dl.async_task` in [techniques/ilp](../techniques/ilp.md)).

### Core Ratio

On 910B2/910B3, the typical hardware configuration is:
- 1 Vector core per 2 Cube cores (in terms of compute resources)
- For Triton scheduling, both are accessed via the same AI Core program model

## Grid Semantics: Physical Core Binding

Ascend grid semantics differ fundamentally from GPU (CUDA) grid semantics:

| Dimension | CUDA/GPU | Ascend |
|-----------|---------|--------|
| Grid meaning | Logical task dimension | Physical core group |
| Hardware mapping | Auto-mapped by driver | Must match physical topology |
| Grid size limit | Very large (millions of blocks) | ≤ Total AI Core count |
| 2D grid | Logical (rows × cols) | Must match physical topology |
| Current kernel limit | Platform-dependent | ≤ 65535 cores |

**Key implication**: On Ascend, you cannot launch more grid blocks than there are physical AI Cores. Attempting to do so requires the `TRITON_ALL_BLOCKS_PARALLEL` environment variable or multi-round scheduling.

### 2D Grid Constraint

2D grids must match the physical core layout of the device. Do not assume arbitrary (M, N) grid shapes are valid. Use 1D grids and compute 2D indices manually in the kernel:

```python
# Safe: 1D grid, compute 2D indices in kernel
pid = tl.program_id(axis=0)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

## On-Chip Memory (UB)

**UB = Unified Buffer**: the on-chip SRAM for each AI Core.

| Hardware | UB Capacity per Core |
|---------|---------------------|
| Atlas 800T/I A2 (910B2/B3) | **192 KB** |

All tile data for a single kernel iteration must fit within 192KB:

```
Tile footprint = sum of all live tensor tile sizes
               = BLOCK_M * BLOCK_K * 2B (fp16 A)
               + BLOCK_K * BLOCK_N * 2B (fp16 B)
               + BLOCK_M * BLOCK_N * 4B (fp32 acc)
               ≤ 192KB
```

Example for `BLOCK_M=128, BLOCK_K=256, BLOCK_N=128, fp16`:
```
A tile: 128 * 256 * 2 = 65,536B = 64KB
B tile: 256 * 128 * 2 = 65,536B = 64KB
acc:    128 * 128 * 4 = 65,536B = 64KB
Total:  192KB (exactly at limit — leave margin in practice)
```

## HBM Bandwidth

- **Theoretical peak**: ~1.8 TB/s (shared across all cores)
- **Practical**: bandwidth is shared when multiple MTE2/MTE3 ops run simultaneously
- For memory-bound roofline: `Time = Data_volume / 1.8e12`

## Core Detection at Runtime

```python
import torch
from triton.runtime import driver

device = torch.cuda.current_device()
props = driver.active.utils.get_device_properties(device)
num_sms = props["multiprocessor_count"]    # number of AI Cores
```

Use `num_sms` to set the persistent kernel grid size or compute optimal tile counts.

## Compilation Optimization Options

These options are passed via `dl.compile_hint()` or environment variables:

| Option | Purpose | Default |
|--------|---------|---------|
| `multibuffer` | Pipeline parallel data movement (double-buffer) | true |
| `unit_flag` | Cube data output optimization (format selection) | None |
| `enable_hivm_auto_cv_balance` | Automatic Cube/Vector balance optimization | None |
| `tile_mix_vector_loop` | Vector core split granularity for mixed tiles | None |
| `tile_mix_cube_loop` | Cube core split granularity for mixed tiles | None |
| `set_workspace_multibuffer N` | Multi-level prefetching (N = buffer depth) | None |
| `auto_blockify_size` | Maps to TRITON_ALL_BLOCKS_PARALLEL | 1 |

### Usage

```python
import triton_ascend.language as dl

@triton.jit
def my_kernel(...):
    dl.compile_hint("multibuffer=true")
    dl.compile_hint("enable_hivm_auto_cv_balance")
    dl.compile_hint("tile_mix_vector_loop=4")
    dl.compile_hint("tile_mix_cube_loop=2")
    dl.compile_hint("set_workspace_multibuffer 4")
    # ... kernel body
```

### `dot_pad_only_k` Hint

For Cube operations where padding is only needed along the K dimension:

```python
dl.compile_hint("dot_pad_only_k")
```

## Environment Variables

| Variable | Purpose | Values |
|----------|---------|--------|
| `TRITON_ALL_BLOCKS_PARALLEL` | Submit all blocks in one scheduling round | 0 (disabled), 1 (enabled) |

```bash
# Enable all-blocks-parallel scheduling (avoids multi-round scheduling overhead)
export TRITON_ALL_BLOCKS_PARALLEL=1
python3 my_kernel_test.py
```

When `TRITON_ALL_BLOCKS_PARALLEL=1`, all grid blocks are submitted to the hardware in a single round. Without it, blocks may be submitted in multiple rounds, adding scheduling latency for large grids.

## 910B2-Specific Features

### `dl.grouped_launch_diagonal`

Hardware-level diagonal PID mapping API available on 910B2:

```python
import triton_ascend.language as dl

pid_m, pid_n = dl.grouped_launch_diagonal(
    pid,       # linear program ID
    num_pid_m, # total tiles in M dimension
    num_pid_n, # total tiles in N dimension
    GROUP_SIZE # tiles per diagonal group
)
```

This performs the PID → diagonal (m, n) remapping at hardware level, avoiding software overhead.

### Data Format Constraints for Dot API

The Dot API (tl.dot / Cube unit) requires specific input/output tensor formats on 910B2. Specifically:
- Input matrices must be in the correct fractal/ND format expected by the Cube unit
- The compiler handles format conversion automatically when using standard tl.dot
- Custom format manipulation requires using the Cube API directly (advanced usage)

**Note**: `tl.make_block_ptr` with `order=(1, 0)` (row-major) is the standard format for Dot API compatibility.

## Pipeline Architecture

The AI Core pipeline:

```
HBM → [MTE2: Load] → UB ← [Vector: Compute] → UB → [MTE3: Store] → HBM
                                    ↑
                              [MTE1: Weight Load from L2]
```

- **MTE2**: Load from HBM or L2 to UB (external reads)
- **MTE3**: Store from UB to HBM (external writes)
- **MTE1**: Load from L2/global memory to Cube input buffers
- **Vector**: Elementwise and reduction operations on UB
- **Cube**: Matrix multiply using Cube input/output buffers

These stages can pipeline (multi-buffer) when data dependencies allow.

## 相关页面
- [techniques/tiling](../techniques/tiling.md) — UB capacity constraints and tile sizing
- [techniques/ilp](../techniques/ilp.md) — Pipeline stages and how to overlap them
- [techniques/core_mapping](../techniques/core_mapping.md) — Grid constraints and diagonal mapping
- [workflows/profiling](../workflows/profiling.md) — Profiling metrics mapped to pipeline stages
- [api/triton_api_cheatsheet](../api/triton_api_cheatsheet.md) — dl.compile_hint and environment variable reference
