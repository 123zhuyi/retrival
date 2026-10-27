---
title: End-to-End Optimization Workflow
category: workflows
tags: [optimization, workflow, bottleneck, routing, experience-index, profiling]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# End-to-End Optimization Workflow

## Overall Flow

```
Profile → Identify bottleneck → Route to experiences → Implement → Re-profile → Validate
```

Each step is described in detail below. The goal is to systematically close the gap between measured performance and theoretical hardware limits.

## Step 1: Pattern Detection (Which Ops/Pages to Use)

Before optimizing, identify the primary compute pattern of the kernel:

| Pattern | Indicator | Primary Reference |
|---------|-----------|------------------|
| Matrix multiply | `tl.dot(a, b)` in kernel | [ops/matmul](../ops/matmul.md) |
| Flash Attention | `softmax(QK^T / sqrt(d_k)) @ V` | [ops/attention](../ops/attention.md) |
| Elementwise | Per-element, same input/output shape | [ops/elementwise](../ops/elementwise.md) |
| Normalization | mean/var/rms computation over rows | [ops/normalization](../ops/normalization.md) |
| Reduction | `tl.sum / tl.max` along axis | [ops/reduction](../ops/reduction.md) |
| Gather/Scatter | Index-based `tl.gather / tl.scatter` | [ops/gather_scatter](../ops/gather_scatter.md) |

Mixed patterns (e.g., GEMM followed by LayerNorm) should start with the dominant compute pattern and apply fusion techniques.

## Step 2: Routing by Bottleneck

After profiling (see [workflows/profiling](profiling.md)), identify the primary bottleneck and route to the appropriate technique:

| Bottleneck | Symptom | Solution |
|-----------|---------|---------|
| Memory bandwidth | `aiv_mte2_time` high (>2x theoretical) | [techniques/tiling](../techniques/tiling.md) (increase tile size), [techniques/bank_conflict](../techniques/bank_conflict.md) (alignment), [techniques/persistent_kernels](../techniques/persistent_kernels.md) |
| Scalar fallback | `aiv_scalar_ratio` > 10% | [techniques/data_types](../techniques/data_types.md): cast int64/int32 to fp32 |
| L2 bank conflict | Performance degrades with large M, N; `aiv_mte2_time` >> theoretical | [techniques/core_mapping](../techniques/core_mapping.md): switch to diagonal mapping |
| Grid overhead | Many short-lived blocks; high kernel launch count | [techniques/persistent_kernels](../techniques/persistent_kernels.md): fix grid to num_aicores |
| Pipeline stall | Simulator shows gap between MTE2 and Vector | [techniques/ilp](../techniques/ilp.md): `care_padding=False`, for-loop tiling |
| Numerical instability | NaN/Inf in output; incorrect results | [techniques/online_algorithms](../techniques/online_algorithms.md), [techniques/data_types](../techniques/data_types.md) |
| Low Cube utilization | `aiv_cube_time` low for matmul | [techniques/tiling](../techniques/tiling.md): increase BLOCK_M/BLOCK_N to 128+ |
| Fusion opportunity | Consecutive kernels with intermediate HBM writes | [techniques/operator_fusion](../techniques/operator_fusion.md) |

## Step 3: Experience Routing Table

Match the workload to the most relevant expert experience entry:

| Task Type | Primary Experience | Expected Boost | Key Technique |
|-----------|-------------------|---------------|--------------|
| Large matmul (M, N, K ≥ 256) | exp_matmul_001 | +40% | L0 128×128×128 tiling |
| Matmul M, N ≥ 512 (square) | exp_matmul_diag_tiling_001 | +25% | Diagonal core mapping |
| Flash Attention (seq ≥ 1024) | exp_fused_attention_002 | +50% | Online softmax, O(S) memory |
| RMSNorm large hidden dim | exp_rmsnorm_npu_001 | +35% | Column blocking, two-pass |
| Split-K reduction (K ≥ 4096) | exp_splitk_reduction_001 | +35% | Atomic partial sums |
| MoE expert routing | exp_gs_bin_003 | +35% | Binned expert dispatch |
| Elementwise activation | exp_elem_fusion_001 | +35% | Inline activation into matmul |

These experiences are detailed in the EXPERIENCE_INDEX with full code examples and configuration parameters.

## Step 4: Implementation Checklist

After selecting the technique/experience, verify these implementation requirements:

### Tiling
- [ ] BLOCK_M, BLOCK_N ∈ {128, 256} for matmul
- [ ] BLOCK_K = 256 for matmul
- [ ] BLOCK_SIZE ∈ {128, 256, 512, 1024, 2048, 4096} for vectorized ops
- [ ] Tile footprint fits in 192KB UB
- [ ] Row widths are 512B-aligned

### Core Mapping
- [ ] M ≥ 512 AND N ≥ 512 → use `dl.grouped_launch_diagonal`
- [ ] AutoTune includes ≥ 3 configs with different GROUP_SIZE (8, 16, 32)
- [ ] No `tl.swizzle2d` usage

### Data Types
- [ ] No int64/int32 in CMP or ADD on Vector unit
- [ ] `tl.arange` results cast to fp32 before comparison
- [ ] Mixed precision: fp32 accumulation for fp16/bf16 inputs

### ILP
- [ ] `care_padding=False` on masked loads where safe
- [ ] For-loop tiling when total_elems > BLOCK_SIZE
- [ ] `dl.async_task` for fused Cube+Vector kernels

### Memory
- [ ] `tl.make_block_ptr` with `order=(1, 0)` for matmul
- [ ] Persistent kernel when n_rows >> num_aicores

## Step 5: Re-profiling Validation

After implementing optimizations:

1. Run the kernel with `msprof` to collect new metrics
2. Compare against pre-optimization baseline:

| Metric | Pre-opt | Post-opt | Target |
|--------|---------|---------|--------|
| `total_time` | - | - | < theoretical × 1.2 |
| `aiv_scalar_ratio` | - | - | < 10% |
| `aiv_mte2_time` | - | - | Close to theoretical |
| `aiv_vec_ratio` | - | - | > 60% (elementwise) |

3. If theoretical performance is not reached, re-run Step 2 to check for secondary bottlenecks.

## Excluded Patterns

The following patterns are too complex, non-standard, or business-specific to be covered in this knowledge base:

| Pattern | Reason Excluded |
|---------|----------------|
| HSTU attention | Non-standard attention semantics |
| Grouped attention with special routing | Business-specific implementation |
| Paged KV cache | Business-specific memory management (PagedAttention, vLLM) |
| MoE routing with dynamic expert selection | Highly dynamic, hardware-specific routing logic |

For these patterns, consult domain-specific implementations and apply the general techniques from this knowledge base as appropriate.

## Quick-Start Decision Tree

```
Start: I have a slow kernel
│
├─ Do I know the bottleneck?
│   ├─ No → Run msprof → Check metrics → Continue below
│   └─ Yes → Skip to routing
│
├─ aiv_scalar_ratio > 10%?
│   └─ Yes → data_types.md: cast int64 to fp32
│
├─ aiv_mte2_time > 2x theoretical?
│   ├─ Matmul with M,N ≥ 512 → core_mapping.md: diagonal
│   ├─ Any op → tiling.md: check alignment, increase BLOCK_SIZE
│   └─ Still high → persistent_kernels.md
│
├─ aiv_vec_ratio < 30%?
│   ├─ Elementwise → tiling.md: increase BLOCK_SIZE
│   └─ Has load/compute dependency → ilp.md: care_padding, for-loop
│
├─ Pipeline stall in simulator?
│   └─ Yes → ilp.md: multibuffer, dl.async_task
│
├─ Good single-op perf but two ops are slow together?
│   └─ Yes → operator_fusion.md: fuse into single kernel
│
└─ Near theoretical? → Validate correctness → Done
```

## 相关页面
- [workflows/profiling](profiling.md) — Detailed profiling commands and metric interpretation
- [techniques/tiling](../techniques/tiling.md) — Tiling strategies reference
- [techniques/core_mapping](../techniques/core_mapping.md) — Core mapping and diagonal selection
- [techniques/data_types](../techniques/data_types.md) — Scalar fallback fixes
- [techniques/ilp](../techniques/ilp.md) — Pipeline optimization
- [techniques/operator_fusion](../techniques/operator_fusion.md) — Fusion patterns and speedups
- [techniques/persistent_kernels](../techniques/persistent_kernels.md) — Grid-stride persistent pattern
- [hardware/ascend_910b](../hardware/ascend_910b.md) — Hardware constraints reference
