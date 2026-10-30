# Abs Operator Knowledge Extraction Report

**Date**: 2026-07-27
**Task**: Extract Abs operator knowledge from ops-math `math/abs` into Triton-Ascend KernelWiki
**Branch**: `feat/kb-ops-math-abs`
**Source commit**: `b56b2db0ebc3fcc96f5f4c7f3060b4375481175e`

## Summary

The CANN ops-math Abs operator has been fully analyzed and its knowledge extracted into the Triton-Ascend KernelWiki. The operator is a straightforward elementwise absolute value operation with special handling for BF16 (FP32 intermediate) and complex dtypes (numerical stability).

## Files Analyzed

| # | File | Role | Key Findings |
|---|------|------|--------------|
| 1 | `README.md` | Overview | Operator spec, product matrix |
| 2 | `docs/aclnnAbs.md` | API docs | Two-phase API, dtype/format constraints |
| 3 | `op_api/abs.h` | API decl | Level0 Abs declaration |
| 4 | `op_api/abs.cpp` | API impl | AICore/AICPU dispatch, dtype promotion |
| 5 | `op_api/aclnn_abs.h` | API decl | Level2 aclnnAbs declaration |
| 6 | `op_api/aclnn_abs.cpp` | API impl | Contiguous→compute→ViewCopy pipeline |
| 7 | `op_graph/abs_proto.h` | Graph IR | GE graph registration |
| 8 | `op_host/abs_def.cpp` | OpDef | AICore config, binary file binding |
| 9 | `op_host/abs_infershape.cpp` | Shape infer | Shape pass-through, dtype promotion |
| 10 | `op_host/arch35/abs_tiling_arch35.h` | Tiling header | Tiling class declaration |
| 11 | `op_host/arch35/abs_tiling_arch35.cpp` | Tiling impl | **Tiling keys 101/102/103**, per-dtype DAG dispatch |
| 12 | `op_host/config/ascend950/abs_binary.json` | Binary config | 9 binary variants |
| 13 | `op_host/config/ascend950/abs_simplified_key.ini` | Config | Simplified key mode |
| 14 | `op_kernel/abs_apt.cpp` | Kernel entry | **AIV-only**, tiling key dispatch |
| 15 | `op_kernel/abs_struct.h` | Struct | `AbsTilingData` |
| 16 | `op_kernel/arch35/abs_dag.h` | DAG | **Real dtype pipeline** |
| 17 | `op_kernel/arch35/abs_complex_dag.h` | DAG | **Complex abs algorithm** |
| 18 | `CMakeLists.txt` | Build | Build config |
| 19 | `examples/test_aclnn_abs.cpp` | Example | Usage example |
| 20 | `examples/test_geir_abs.cpp` | Example | Graph mode example |
| 21 | `tests/assets/golden.py` | Golden | NumPy reference |
| 22 | `tests/ut/op_api/test_aclnn_abs.cpp` | UT | API unit test |
| 23 | `tests/ut/op_host/test_abs_infershape.cpp` | UT | Shape inference test |
| 24 | `tests/ut/op_host/arch35/test_abs_tiling_arch35.cpp` | UT | Tiling test |

## Files Created/Modified

### New files

| Path | Purpose |
|------|---------|
| `kernelwiki/sources/docs/cann-ops-math-abs-source.md` | Comprehensive source document |
| `kernelwiki/wiki/kernels/elementwise-abs.md` | KernelWiki knowledge page |
| `kernelwiki/artifacts/elementwise-abs/PROVENANCE.yaml` | Provenance metadata |
| `kernelwiki/artifacts/elementwise-abs/source_file_inventory.md` | Source file inventory |
| `kernelwiki/artifacts/elementwise-abs/tiling_dispatch_tables.md` | Tiling/dispatch tables |
| `kernelwiki/artifacts/elementwise-abs/abs_triton_ascend_skeleton.py` | Triton skeleton code |
| `kernelwiki/artifacts/elementwise-abs/codegen_constraints.yaml` | Code generation constraints |
| `reports/abs_extraction_report.md` | This report |
| `reports/abs_knowledge_usage_audit.md` | Knowledge usage audit |

## Abs Operator Semantic Summary

```
y = abs(x)
```

- Element-wise absolute value
- Input: any real or complex tensor (0D-8D, ND format)
- Output: same shape as input
- Real: same dtype as input (except BF16 uses FP32 intermediate)
- Complex: promoted output (COMPLEX32→FLOAT16, COMPLEX64→FLOAT)
- AIV-only execution
- Three tiling keys: 101 (BF16), 102 (other real), 103 (complex)

## CANN-to-Triton-Ascend Mapping

| CANN Concept | Triton-Ascend |
|--------------|---------------|
| TPipe / ElementwiseSch | `@triton.jit` kernel with load/compute/store |
| Vec::CopyIn | `tl.load` with tail mask |
| Vec::Cast | `.to()` dtype conversion |
| Vec::Abs | `tl.abs(x)` or `tl.where(x < 0, -x, x)` |
| Vec::CopyOut | `tl.store` with tail mask |
| EleBaseTilingData.blockNum | Grid size = `(n + BLOCK_SIZE - 1) // BLOCK_SIZE` |
| KERNEL_TYPE_AIV_ONLY | Default Triton Ascend vector dispatch |
| 16 MiB workspace | Not needed in Triton |
| Complex interleaved | Manual de-interleave load |

## Key Design Decisions Captured

1. **Three tiling keys** map to dtype families, not individual dtypes
2. **BF16 must use FP32 intermediate** — the Cast nodes in the DAG are not optional
3. **Complex64 abs requires numerical stability** — the CANN kernel implements a sophisticated normalization algorithm that a Triton implementation should match
4. **Complex32 abs can use simpler `sqrt(real² + imag²)`** — lower precision makes the simpler path acceptable
5. **AIV-only is the standard** for elementwise vector math on Ascend
6. **16 MiB workspace is a CANN internal detail** — Triton manages buffers differently