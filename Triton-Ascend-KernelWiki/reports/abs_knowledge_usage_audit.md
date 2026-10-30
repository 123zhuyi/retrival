# Abs Knowledge Usage Audit

**Date**: 2026-07-27
**Purpose**: Audit each knowledge item extracted from CANN ops-math Abs to determine whether and how it was used in the Triton-Ascend abstraction.

## Audit Table

| # | Knowledge Item | Source File | Use Type | Manifestation | Rationale |
|---|---------------|-------------|----------|---------------|-----------|
| 1 | `y = abs(x)` semantic | `README.md`, `docs/aclnnAbs.md` | **direct** | Skeleton `tl.abs(x)`; kernel page semantic | This is the core mathematical operation, maps directly |
| 2 | Input shape → output shape | `abs_infershape.cpp` | **direct** | Kernel page: "Output shape equals input shape"; constraints: `output shape == input shape` | Fundamental elementwise identity |
| 3 | ND format, 0-8D | `aclnn_abs.h:21`, `aclnn_abs.cpp:110-127` | **direct** | Constraints doc: `max rank = 8`, `data format = ND only` | Direct constraint carry-over |
| 4 | Real dtype: same in/out | `abs_tiling_arch35.cpp:61-66` | **direct** | Kernel page dtype table; skeleton for real types | Direct mapping |
| 5 | Complex dtype: promoted output | `abs_infershape.cpp:22-34`, `abs.cpp:97-105` | **direct** | Kernel page complex dtype table; skeleton complex kernel | Output dtype differs from input — must be explicit |
| 6 | COMPLEX64: `sqrt(real² + imag²)` | `abs_complex_dag.h` | **direct** | Skeleton complex kernel: `tl.sqrt(real*real + imag*imag)` | Core formula |
| 7 | Complex64 numerical stability | `abs_complex_dag.h:114-159` | **adapted** | Constraints doc: "needs numerical stability for large/small values" | Triton's `tl.sqrt` may handle some of this natively, but the CANN algorithm is documented for reference |
| 8 | Complex32 simpler path | `abs_complex_dag.h:103-113` | **direct** | Constraints doc: "can use simpler sqrt(sum_sq) path" | Lower precision makes this acceptable |
| 9 | Tiling key 101: BF16 | `abs_tiling_arch35.cpp:30-33` | **direct** | Tiling dispatch table; skeleton `abs_kernel_bf16` | BF16 needs special handling |
| 10 | Tiling key 102: other real | `abs_tiling_arch35.cpp:30-33` | **direct** | Tiling dispatch table; skeleton `abs_kernel_real` | General real-dtype path |
| 11 | Tiling key 103: complex | `abs_tiling_arch35.cpp:30-33` | **direct** | Tiling dispatch table; skeleton `abs_kernel_complex` | Complex path |
| 12 | BF16 → FP32 intermediate → BF16 | `abs_dag.h:26-36` | **direct** | Skeleton `abs_kernel_bf16` with explicit cast; constraints: "BF16 always needs FP32 intermediate" | Critical for precision |
| 13 | DAG: CopyIn → Cast → Abs → Cast → CopyOut | `abs_dag.h` | **adapted** | Skeleton sequence: `tl.load` → `.to()` → `tl.abs` → `.to()` → `tl.store` | CANN DAG → sequential Triton ops |
| 14 | Complex DAG: CopyIn → AbscomplexCustom → CopyOut | `abs_complex_dag.h` | **adapted** | Skeleton complex kernel: interleaved load → compute → store | Custom compute becomes manual Triton code |
| 15 | Real DAG: Abs<dtype> | `abs_dag.h` | **direct** | Skeleton: `tl.abs(x)` | Direct mapping |
| 16 | AIV-only execution | `abs_apt.cpp:27` | **adapted** | Kernel page: "AIV-only is the standard"; constraints: "Kernel type = AIV-only" | CANN explicit marker → Triton implicit (default vector dispatch) |
| 17 | `blockDim = tiling.blockNum` | `abs_tiling_arch35.cpp:47` | **adapted** | Skeleton: `grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)` | CANN blockNum maps to Triton grid size |
| 18 | `coreNum = GetCoreNumAiv()` | `abs_tiling_arch35.cpp:145` | **rejected** | Not used in skeleton | Triton grid is data-driven, not core-count-driven |
| 19 | UB size query | `abs_tiling_arch35.cpp:146` | **rejected** | Not used in skeleton | Triton compiler manages UB allocation |
| 20 | Workspace 16 MiB reserve | `abs_tiling_arch35.cpp:33` | **rejected** | Noted in constraints as "Not needed in Triton" | Triton manages temporary buffers internally |
| 21 | Non-contiguous → Contiguous → ViewCopy | `aclnn_abs.cpp:191,211` | **adapted** | Constraints: "non-contiguous support = Yes" with pointer+stride note | Triton `tl.load` handles arbitrary strides directly |
| 22 | Empty tensor early return | `aclnn_abs.cpp:184-188` | **rejected** | Noted in constraints | Triton kernel with `grid=(0,)` handles this implicitly |
| 23 | UINT8/BOOL passthrough | `aclnn_abs.cpp:73-80,195-199` | **direct** | Constraints: "UINT8/BOOL are identity operations" | Knowledge that these are no-ops |
| 24 | Platform-specific dtype support | `abs.cpp:25-52`, `aclnn_abs.cpp:31-43` | **direct** | Platform support matrix in tiling tables and constraints | Critical for codegen correctness |
| 25 | Tail mask protection | (derived from technique-tail-mask) | **direct** | Skeleton: `mask = offsets < n_elements` on all loads/stores | Elementwise pattern |
| 26 | `REG_OP(Abs)` proto | `abs_proto.h` | **rejected** | Not used in Triton | Graph IR registration is CANN-specific |
| 27 | AICore vs AICPU dispatch | `abs.cpp:107-109` | **rejected** | Not used in Triton | Triton always runs on AICore (vector) |
| 28 | Simplified key mode | `abs_simplified_key.ini` | **rejected** | Not used in Triton | Compiler tooling detail |
| 29 | Binary config per dtype | `abs_binary.json` | **rejected** | Not used in Triton | CANN compilation artifact |
| 30 | Golden test: `np.abs(x)` | `golden.py` | **direct** | Used as correctness reference for skeleton | Reference implementation |

## Summary

| Use Type | Count | Percentage |
|----------|-------|------------|
| **direct** | 14 | 46.7% |
| **adapted** | 7 | 23.3% |
| **rejected** | 9 | 30.0% |

### Observations

1. **Direct uses (46.7%)** dominate — most of the operator's core semantics, dtype rules, and tiling dispatch map cleanly to Triton
2. **Adapted uses (23.3%)** — CANN's DAG pipeline, blockNum, AIV-only marker, and non-contiguous handling all have natural Triton equivalents
3. **Rejected uses (30.0%)** — mostly CANN infrastructure details (workspace, core count, UB size, binary config, graph IR registration) that have no Triton equivalent or are managed automatically by the Triton compiler
4. The **tiling key system** (101/102/103) is the most important structural knowledge — it groups dtypes into families with shared behavior, which directly informs Triton codegen dispatch
5. The **BF16 FP32 intermediate** and **complex numerical stability** are the two most critical non-obvious details that must be preserved in any Triton implementation