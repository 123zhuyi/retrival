---
title: Profiling Methodology for Triton-Ascend Kernels
category: workflows
tags: [profiling, msprof, performance, metrics, bottleneck, simulator]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Profiling Methodology for Triton-Ascend Kernels

## Overview

Profiling is the essential first step in optimization. On Ascend, two tools are available: `msprof` for on-board hardware profiling and the simulator for pipeline visualization. Both produce structured output that maps directly to kernel bottlenecks.

## msProf On-Board Profiling

On-board profiling runs the kernel on real hardware and collects actual timing data.

### Command

```bash
msprof op --kernel-name=target_kernel --output=$HOME/output python3 test.py
```

### Options

| Option | Description |
|--------|-------------|
| `--kernel-name=<name>` | Filter to profile only the specified kernel (by JIT function name) |
| `--output=<dir>` | Output directory for profiling artifacts |
| `--duration=<ms>` | Profile for specified duration |

### Outputs

| File | Contents |
|------|---------|
| `PipeUtilization.csv` | Per-stage pipeline utilization (MTE2, Vector, Cube, MTE3) |
| `op_summary_*.csv` | Per-kernel summary metrics (aiv_vec_ratio, aiv_scalar_ratio, etc.) |

### Minimal Test Script Pattern

```python
import torch
import triton

def run_kernel():
    # Setup inputs
    A = torch.randn(M, K, dtype=torch.float16, device='npu')
    B = torch.randn(K, N, dtype=torch.float16, device='npu')

    # Warm up (not profiled)
    for _ in range(5):
        C = my_kernel(A, B)
    torch.cuda.synchronize()

    # Profiled run
    for _ in range(100):
        C = my_kernel(A, B)
    torch.cuda.synchronize()

run_kernel()
```

```bash
msprof op --kernel-name=my_matmul_kernel --output=$HOME/prof_output python3 test.py
```

## Simulator Pipeline Visualization

The simulator provides instruction-level pipeline visualization without requiring real hardware. It generates a trace that can be opened in Chrome DevTools or MindStudio Insight.

### Command

```bash
export LD_LIBRARY_PATH=/root/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/{soc-version}/lib:$LD_LIBRARY_PATH

msprof op simulator \
    --kernel-name=_layer_norm_fwd_fused \
    --soc-version=Ascend910B3 \
    python3 test.py
```

Replace `{soc-version}` with the target hardware version (e.g., `Ascend910B3`).

### Supported SoC Versions

Ascend series supported by the simulator:
- `Ascend910A`, `Ascend910B`, `Ascend910B1`, `Ascend910B2`, `Ascend910B2C`, `Ascend910B3`, `Ascend910B4`
- `Ascend310`, `Ascend310P`, `Ascend310B` series

### Simulator Outputs

| File | Contents | Viewer |
|------|---------|--------|
| `trace.json` | Chrome DevTools trace format | `chrome://tracing` or MindStudio Insight |
| `visualize_data.bin` | Binary pipeline visualization data | MindStudio Insight |

To open `trace.json`:
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select `trace.json`

The trace shows the exact sequence of MTE2, Vector, Cube, and MTE3 instructions and their timing, allowing you to identify:
- Pipeline stalls (gaps between stages)
- Serialization (stages not overlapping)
- Hot spots (stages with disproportionate time)

## Key Metrics from `op_summary_*.csv`

| Metric | Meaning | Ideal Value |
|--------|---------|-------------|
| `aiv_vec_ratio` | Fraction of time on Vector compute unit | High (>60% for elementwise ops) |
| `aiv_scalar_ratio` | Fraction of time on Scalar unit (fallback) | Low (<10%) |
| `aiv_mte2_time` | Load latency (HBM→UB transfer time) | Low (close to theoretical) |
| `aiv_mte3_time` | Store latency (UB→HBM transfer time) | Low |
| `aiv_cube_time` | Cube unit execution time | High (for matmul kernels) |
| `aiv_vec_time` | Vector unit execution time | High (for elementwise kernels) |
| `total_time` | Total kernel wall-clock time | Minimize |

## Theoretical Performance Baseline

Compute the theoretical lower bound before profiling to understand how far from optimal the kernel is.

### Memory-Bound Kernel

For kernels where performance is limited by HBM bandwidth:

```
Memory_time = Data_volume / HBM_bandwidth
            = (input_bytes + output_bytes) / 1.8e12

Example: 4096×4096 fp16 elementwise
Data_volume = 4096 * 4096 * 2B * 2 (read + write) = 67,108,864B ≈ 64MB
Memory_time = 64MB / 1.8TB/s = 35.6 μs
```

### Compute-Bound Kernel (Matmul)

For kernels limited by Cube unit FLOPS:

```
Compute_time = FLOPs / Theoretical_FLOPS
             = (2 * M * N * K) / peak_TFLOPS

Example: 4096×4096×4096 fp16 matmul
FLOPs = 2 * 4096^3 = 137.4 GFLOP
Compute_time = 137.4 GFLOP / (peak TFLOPS) 

For memory bound:
Data_volume = (M*K + K*N + M*N) * 2B = (4096^2 * 3) * 2 ≈ 96MB
Memory_time = 96MB / 1.8TB/s = 53.3 μs → actual is ~37.28 μs with L2 reuse
```

The kernel is compute-bound if `Compute_time > Memory_time`, memory-bound otherwise.

## Bottleneck Identification

### 1. Scalar Fallback (`aiv_scalar_ratio > 10%`)

**Symptom**: `aiv_scalar_ratio` is high; Vector unit time is low.

**Cause**: Operations on int64 or int32 types that the Vector unit does not support.

**Fix**: See [techniques/data_types](../techniques/data_types.md) — cast to fp32 before comparison or arithmetic.

```python
# Common fix
cols = tl.arange(0, BLOCK_N).to(tl.float32)  # was int64
mask = cols < N                                # now fp32 CMP → Vector unit
```

### 2. Memory Bottleneck (`aiv_mte2_time` high)

**Symptom**: `aiv_mte2_time` is much higher than theoretical; Vector/Cube utilization is low.

**Causes and fixes**:
| Sub-cause | Fix |
|-----------|-----|
| L2 bank conflict | Diagonal core mapping (see [techniques/core_mapping](../techniques/core_mapping.md)) |
| Tile too small → too many small transfers | Increase BLOCK_SIZE |
| Unaligned access | Ensure 512B row alignment (see [techniques/tiling](../techniques/tiling.md)) |
| Poor spatial locality | Adjust core mapping strategy |

### 3. Low Vector Utilization (`aiv_vec_ratio` low)

**Symptom**: `aiv_vec_ratio` < 30% for an elementwise kernel.

**Causes and fixes**:
| Sub-cause | Fix |
|-----------|-----|
| BLOCK_SIZE too small | Increase to 512 or 1024 |
| Scattered memory access | Use contiguous indexing, add padding |
| Type mismatch → scalar | Check data types |
| Pipeline stall (load before compute) | Add `care_padding=False`, use for-loop tiling |

### 4. Pipeline Stall (Load/Compute Dependency)

**Symptom**: Simulator trace shows gap between MTE2 and Vector stages.

**Fix**: See [techniques/ilp](../techniques/ilp.md) — use `care_padding=False` and for-loop tiling to enable multi-buffer.

### 5. Large Gap from Theoretical (Tiling Overhead)

**Symptom**: Measured time is 3-5x theoretical for simple ops.

**Causes and fixes**:
| Sub-cause | Fix |
|-----------|-----|
| Too many small tiles → kernel launch overhead | Reduce tile count; use persistent kernel |
| Grid too large → multi-round scheduling | Set `TRITON_ALL_BLOCKS_PARALLEL=1` |
| Autotune overhead | Cache autotune results |

## Complete Profiling Workflow

```
1. Run kernel without profiling → record baseline time
2. Compute theoretical time (memory bound or compute bound)
3. Compute performance ratio = theoretical / measured
4. If ratio < 0.5 (>2x from theoretical): profile with msprof
5. Check aiv_scalar_ratio first (cheapest fix)
6. Check aiv_mte2_time vs theoretical (bank conflict / alignment)
7. Check aiv_vec_ratio (vectorization health)
8. Use simulator for instruction-level pipeline analysis
9. Apply fix from experience table
10. Re-profile to confirm improvement
11. Repeat until ratio > 0.8 (within 20% of theoretical)
```

## 相关页面
- [workflows/optimization_flow](optimization_flow.md) — End-to-end optimization decision tree
- [techniques/data_types](../techniques/data_types.md) — Fixing aiv_scalar_ratio issues
- [techniques/bank_conflict](../techniques/bank_conflict.md) — Fixing aiv_mte2_time issues
- [techniques/ilp](../techniques/ilp.md) — Fixing pipeline stalls
- [techniques/tiling](../techniques/tiling.md) — Fixing tile sizing issues
- [hardware/ascend_910b](../hardware/ascend_910b.md) — Hardware pipeline architecture reference
