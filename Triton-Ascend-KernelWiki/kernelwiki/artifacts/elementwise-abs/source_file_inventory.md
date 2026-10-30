# Abs Operator Source File Inventory

> Repository: ops-math / math/abs
> Commit: b56b2db0ebc3fcc96f5f4c7f3060b4375481175e
> Retrieved: 2026-07-27

## Overview

| Category | Count | Key Files |
|----------|-------|-----------|
| API declarations | 2 | `aclnn_abs.h`, `abs.h` |
| API implementations | 2 | `aclnn_abs.cpp`, `abs.cpp` |
| Graph proto | 1 | `abs_proto.h` |
| Host (def/shape/tiling) | 4 | `abs_def.cpp`, `abs_infershape.cpp`, `abs_tiling_arch35.cpp/h` |
| Config | 2 | `abs_binary.json`, `abs_simplified_key.ini` |
| Kernel | 3 | `abs_apt.cpp`, `abs_dag.h`, `abs_complex_dag.h` |
| Kernel struct | 1 | `abs_struct.h` |
| Examples | 2 | `test_aclnn_abs.cpp`, `test_geir_abs.cpp` |
| Tests | 5 | ut/st test files |
| Golden | 1 | `golden.py` |
| Docs | 1 | `aclnnAbs.md` |
| Build | 1 | `CMakeLists.txt` |
| **Total** | **23** | |

## File Details

### 1. README.md
- Operator overview: `y = abs(x)`
- Product support matrix
- Parameter table with dtype and format
- Call examples: aclnn and graph mode

### 2. docs/aclnnAbs.md
- API documentation for `aclnnAbs`
- Two-phase API: `aclnnAbsGetWorkspaceSize` + `aclnnAbs`
- Full parameter descriptions with dtype/format/shape constraints
- Error codes
- Complete C++ example

### 3. op_api/abs.h
- Level0 API declaration: `const aclTensor* Abs(const aclTensor* self, aclOpExecutor* executor)`

### 4. op_api/abs.cpp
- `OP_TYPE_REGISTER(Abs)` — operator type registration
- `IsAiCoreSupport()` — platform-specific AICore dtype support check
- `AbsInferShape()` — output shape = input shape
- `AbsAiCore()` — AICore kernel dispatch via `ADD_TO_LAUNCHER_LIST_AICORE`
- `AbsAiCpu()` — AICPU fallback via `ADD_TO_LAUNCHER_LIST_AICPU`
- `Abs()` — main dispatch: infer shape, allocate output tensor (with complex dtype promotion), route to AICore or AICPU

### 5. op_api/aclnn_abs.h
- Level2 API declaration: `aclnnAbsGetWorkspaceSize` and `aclnnAbs`
- `ACLNN_MAX_SHAPE_RANK = 8`
- Mermaid compute graph: `self → Contiguous → Abs → ViewCopy → out`

### 6. op_api/aclnn_abs.cpp
- Platform-specific dtype support lists (910, 910B, RegBase)
- `COMPLEX_IN_AND_OUT_DTYPE_MAP`: Complex → promoted dtype mapping
- `CheckParams()`: null, dtype, format, shape validation
- `IsSupportComplexAbs()`: platform check for complex abs support (910B+)
- `aclnnAbsGetWorkspaceSize()`: full pipeline — Contiguous → passthrough/ComplexAbs/Abs → ViewCopy
- `aclnnAbs()`: execute via `CommonOpExecutorRun`

### 7. op_graph/abs_proto.h
- GE graph IR registration: `REG_OP(Abs)`
- Input `x`: types `{uint8, bool, bf16, float16, float32, double, int8, int16, int32, int64, complex32, complex64}`
- Output `y`: same type as `x`

### 8. op_host/abs_def.cpp
- `OpDef` registration for `Abs`
- Input `x`: 9 dtypes + 9 ND formats
- Output `y`: 9 dtypes (complex → promoted), 9 ND formats
- AICore config: dynamic compile, dynamic rank, dynamic shape, precision reduce
- Binary file: `abs_apt`

### 9. op_host/abs_infershape.cpp
- `InferDataType4Abs`: COMPLEX64 → FLOAT, COMPLEX32 → FLOAT16, else pass through
- Shape inference: `InferShape4Elewise` (elementwise identity)

### 10. op_host/arch35/abs_tiling_arch35.h
- `AbsCompileInfo`: coreNum, ubSize
- `AbsTiling` class: `RunTiling()`, `CalcOutputDtype()`, `SetTilingData()`

### 11. op_host/arch35/abs_tiling_arch35.cpp
- **Tiling keys**: 101 (BF16), 102 (other real), 103 (complex)
- **DoTiling** per dtype: float16, complex64, complex32, float, bf16, int8/16/32/64
- **Workspace**: 16 MiB reserved
- **TilingPrepare**: queries AIV core count and UB size

### 12. op_host/config/ascend950/abs_binary.json
- 9 binary variants (BF16, FP16, FP32, INT8, INT16, INT32, INT64, COMPLEX64→FP32, COMPLEX32→FP16)
- All ND format, dynamic shape (-2), FormatAgnostic

### 13. op_host/config/ascend950/abs_simplified_key.ini
- Simplified key mode: `default=0`

### 14. op_kernel/abs_apt.cpp
- Kernel entry: `abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)`
- `KERNEL_TYPE_AIV_ONLY`
- Dispatch by tiling key: 101/102/103
- Uses `ElementwiseSch` for pipeline orchestration

### 15. op_kernel/abs_struct.h
- `AbsTilingData { EleBaseTilingData baseTiling }`

### 16. op_kernel/arch35/abs_dag.h
- Real dtype DAG: `CopyIn → Cast → Abs → Cast → CopyOut`
- Template: `AbsDag<U, T>` (U=storage, T=compute)
- BF16 path: `U=bfloat16_t, T=float`

### 17. op_kernel/arch35/abs_complex_dag.h
- Complex dtype DAG: `CopyIn → AbscomplexCustom → CopyOut`
- `AbscomplexDag<U, T>` (U=complex storage, T=promoted output)
- COMPLEX32 path: interleaved load → cast to float → square → sum → sqrt → cast to half
- COMPLEX64 path: de-interleave → sort → normalize → compute → denormalize
- Numerical stability: subnormal handling, overflow prevention

### 18. op_kernel/abs_apt.cpp (kernel entry)
- See above (item 14)

### 19. tests/assets/golden.py
- `abs_golden()`: NumPy reference implementation
- COMPLEX32: split interleaved → `sqrt(real^2 + imag^2)`
- BF16: cast to float32 → `np.abs()` → cast back
- Other: `np.abs(x)` directly

### 20. tests/ut/op_api/test_aclnn_abs.cpp
- Unit test for aclnnAbs API

### 21. tests/ut/op_host/test_abs_infershape.cpp
- Unit test for shape inference

### 22. tests/ut/op_host/arch35/test_abs_tiling_arch35.cpp
- Unit test for arch35 tiling

### 23. CMakeLists.txt
- Build configuration