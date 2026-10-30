---
id: source-cann-ops-math-abs
title: CANN ops-math Abs Operator Source
type: source
tags: [cann, ops-math, abs, elementwise, aiv-only, real-dtype, complex-dtype]
architectures: [Ascend910B1, Ascend950PR, Ascend950DT]
kernel_types: [math]
operators: [abs]
confidence: high
reproducibility: source
sources: []
related: [kernel-elementwise-abs]
---

# CANN ops-math Abs Operator Source

> Extracted from ops-math `math/abs` at commit `b56b2db0ebc3fcc96f5f4c7f3060b4375481175e`.

## Operator Semantics

```
y = abs(x)
```

For each element of the input tensor `x`, compute its absolute value element-wise. Output shape matches input shape exactly.

## Data Type Rules

### Real dtype

| Input dtype | Output dtype | Notes |
|-------------|--------------|-------|
| FLOAT       | FLOAT        | Same dtype in/out |
| FLOAT16     | FLOAT16      | Same dtype in/out |
| BFLOAT16    | BFLOAT16     | Same dtype in/out; intermediate FP32 |
| DOUBLE      | DOUBLE       | Same dtype in/out |
| INT8        | INT8         | Same dtype in/out |
| INT16       | INT16        | Same dtype in/out |
| INT32       | INT32        | Same dtype in/out |
| INT64       | INT64        | Same dtype in/out |
| UINT8       | UINT8        | Passthrough (abs is identity) |
| BOOL        | BOOL         | Passthrough (abs is identity) |

Source: `op_api/aclnn_abs.cpp` lines 90-108, `op_host/arch35/abs_tiling_arch35.cpp` lines 61-81.

### Complex dtype

| Input dtype | Output dtype | Formula |
|-------------|--------------|---------|
| COMPLEX32   | FLOAT16      | `sqrt(real^2 + imag^2)` with numerical stability |
| COMPLEX64   | FLOAT        | `sqrt(real^2 + imag^2)` with numerical stability |

Source: `op_host/abs_infershape.cpp` lines 22-34, `op_api/abs.cpp` lines 97-105.

The output dtype for complex input is promoted: COMPLEX32 → FLOAT16, COMPLEX64 → FLOAT. The `abs_def.cpp` output DataType list explicitly shows this mapping (lines 30-36).

### Platform-specific dtype support

- **Ascend910 (DAV_210)**: FLOAT, FLOAT16, INT32 — *no BF16, INT64, COMPLEX* (`op_api/aclnn_abs.cpp` lines 31-33)
- **Ascend910B (DAV_2201)**: plus BF16, COMPLEX64 (`op_api/aclnn_abs.cpp` lines 35-38)
- **RegBase arch (DAV_310P+)**: plus COMPLEX32, INT8, INT16 (`op_api/aclnn_abs.cpp` lines 40-43)
- **Ascend610Lite (DAV_3102)**: FLOAT, FLOAT16, INT32 only (`op_api/abs.cpp` lines 28-29)

### AICore vs AICPU dispatch

- AICore: FLOAT, FLOAT16, BF16, INT32 (and extended per platform)
- AICPU (fallback): all remaining supported dtypes
- Source: `op_api/abs.cpp` lines 25-52

## Shape / Layout Rules

- Shape: ND format, 0D to 8D supported
- Output shape matches input shape exactly
- Non-contiguous tensors supported (handled via Contiguous → compute → ViewCopy pipeline)
- Source: `op_api/aclnn_abs.h` line 21 (`ACLNN_MAX_SHAPE_RANK = 8`), `op_api/aclnn_abs.cpp` lines 110-127

## Host Tiling Flow

### Tiling Preparation (`op_host/arch35/abs_tiling_arch35.cpp`)

1. **TilingPrepareForAbs** (line 138-148): Query platform info
   - `coreNum = ascendcPlatform.GetCoreNumAiv()` — uses AIV cores only
   - `ubSize = GetCoreMemSize(UB)` — UB (Unified Buffer) size for the platform

2. **CalcOutputDtype** (line 51-81): Validate dtype pairing
   - Real dtype: input == output dtype
   - COMPLEX64 → output must be FLOAT
   - COMPLEX32 → output must be FLOAT16

3. **RunTiling** (line 83-123): Dispatch to ElewiseBaseTiling with dtype-specific DAG
   - For each dtype, calls `elewiseBaseTiling.DoTiling<SpecificDag>()`
   - Sets `tiling->baseTiling` (EleBaseTilingData) with blockDim, blockNum, etc.

4. **SetTilingData** (line 35-49): Set tiling key and blockDim
   - Workspace: reserves 16 MiB (`ABS_WORKSPACE_RESERVE_BYTE = 16777216`)

### Tiling Keys

| Key | Name | Condition | DAG Template |
|-----|------|-----------|--------------|
| 101 | `ELEMENTWISE_BF16` | output dtype == BF16 | `AbsDag<bfloat16_t, float>` (FP32 intermediate) |
| 102 | `ELEMENTWISE_OTHER` | real dtype, not BF16 | `AbsDag<dtype, dtype>` |
| 103 | `ELEMENTWISE_COMPLEX` | COMPLEX32 or COMPLEX64 | `AbscomplexDag<complexX, promoted>` |

Source: `op_host/arch35/abs_tiling_arch35.cpp` lines 30-33, `op_kernel/abs_apt.cpp` lines 33-73.

### blockDim / coreNum / UB

- `blockDim = tiling->baseTiling.blockNum` (determined by ElewiseBaseTiling based on data size and UB capacity)
- `coreNum = GetCoreNumAiv()` — uses AIV cores only (`KERNEL_TYPE_AIV_ONLY` in kernel entry)
- Workspace: 16 MiB reserved (`ABS_WORKSPACE_RESERVE_BYTE`)

Source: `op_host/arch35/abs_tiling_arch35.cpp` lines 37-39, 47-48, 144-148.

## Kernel Dispatch Logic

### Kernel entry (`op_kernel/abs_apt.cpp`)

```cpp
extern "C" __global__ __aicore__ void abs(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` — AIV-only elementwise pipeline
- `REGISTER_TILING_DEFAULT(AbsTilingData)` — register tiling data struct
- `GET_TILING_DATA_WITH_STRUCT(AbsTilingData, tilingData, tiling)` — deserialize tiling

Dispatch by tiling key:
- **Key 101 (BF16)**: `ElementwiseSch<0, AbsDag<bfloat16_t, float>::OpDag>` — BF16 input, FP32 intermediate, BF16 output
- **Key 102 (other real)**: `ElementwiseSch<0, AbsDag<dtype, dtype>::OpDag>` — same dtype throughout
- **Key 103 (complex)**: `ElementwiseSch<0, AbscomplexDag<complexX, promoted>::OpDag>` — complex input, promoted output

### CANN Pipeline (TPipe / ElementwiseSch)

The `ElementwiseSch` provides a load-compute-store pipeline:
1. `sch.Init(x, y)` — bind input/output GM addresses
2. `sch.Process()` — execute the DAG pipeline

This abstracts:
- CopyIn (GM → UB): contiguous vector load
- Cast (optional): dtype conversion in UB
- Compute: vector Abs operation in UB
- Cast (optional): dtype conversion back
- CopyOut (UB → GM): contiguous vector store

### AIV-only Elementwise Pipeline

The kernel is marked `KERNEL_TYPE_AIV_ONLY`, meaning it runs exclusively on AIV (AI Vector) cores, not on AIC (AI Cube) cores. This is standard for elementwise vector math operations.

## Real Dtype DAG

From `op_kernel/arch35/abs_dag.h`:

```
CopyIn<U> → Cast<T, U, 0> → Abs<T> → Cast<U, T, 1> → CopyOut<U>
```

Where:
- `U` = storage dtype (e.g., float, half, int32)
- `T` = compute dtype (default float for BF16, same as U otherwise)

For BF16 (key 101): `CopyIn<bfloat16_t> → Cast<float, bfloat16_t> → Abs<float> → Cast<bfloat16_t, float> → CopyOut<bfloat16_t>`

For other real dtypes (key 102): `CopyIn<dtype> → Cast<dtype, dtype, 0> → Abs<dtype> → Cast<dtype, dtype, 1> → CopyOut<dtype>`
(The Cast nodes are identity when U == T.)

### Abs Vec Operation

The `Vec::Abs<T>` operation computes `|x|` element-wise. For integer types, this is branchless two's complement abs. For floating point, it clears the sign bit.

## Complex Dtype DAG and Numerical Stability

From `op_kernel/arch35/abs_complex_dag.h`:

### DAG structure

```
CopyIn<U> → AbscomplexCustom<T, U> → CopyOut<T>
```

Where:
- `U` = complex storage type (complex64 or complex32)
- `T` = promoted output type (float or half)

### Complex Abs Algorithm

The complex abs formula is: `sqrt(real^2 + imag^2)`

For complex input stored as interleaved `[real0, imag0, real1, imag1, ...]` pairs:

**For COMPLEX32 → FLOAT16** (`complex32` input, line 103-113):
1. Load interleaved real/imag pairs
2. Cast each from `T` (half-like) to `float` for precision
3. `vA1Sq = vA * vA`, `vB1Sq = vB * vB` — square real and imag
4. `vSumSq = vA1Sq + vB1Sq` — sum squares
5. `vS = sqrt(vSumSq)` — compute sqrt
6. Cast result from `float` to `T` (half) with saturation

**For COMPLEX64 → FLOAT** (line 114-159), with numerical stability:
1. De-interleave load: load pairs into two registers
2. `vAbsSrcReg0 = |vSrcReg0|`, `vAbsSrcReg1 = |vSrcReg1|` — absolute values
3. Sort: `vA = max(|real|, |imag|)`, `vB = min(|real|, |imag|)`
4. **Zero handling**: if `vB == 0`, result = `vA` (avoids 0/0)
5. **Subnormal handling**: if `vA < min_normal_float`, scale up by `2^24`
6. Extract exponent factor from `vA`: separate into mantissa + exponent
7. Normalize: `vA1 = (vA >> 23 << 23)`, normalize both to [1, 2) range
8. `vA1Sq = vA1 * vA1`, `vB1Sq = vB1 * vB1` — square normalized values
9. `vSumSq = vA1Sq + vB1Sq`, `vS = sqrt(vSumSq)`
10. `vResult = vS * vA0` — multiply back exponent factor
11. Scale down if subnormal
12. **Select**: if originally subnormal, use scaled-down result

This algorithm avoids overflow for large values and underflow for subnormals by normalizing to [1, 2) range.

## Special Paths

### UINT8 / BOOL
Passthrough via ViewCopy (no computation needed). Source: `op_api/aclnn_abs.cpp` lines 73-80, 195-199.

### Empty Tensor
Returns immediately with workspaceSize=0. Source: `op_api/aclnn_abs.cpp` lines 63-70, 184-188.

### Non-contiguous Input
Handled via `l0op::Contiguous` before compute, `l0op::ViewCopy` after. Source: `op_api/aclnn_abs.cpp` lines 191, 211-212.

## Binary Configuration

From `op_host/config/ascend950/abs_binary.json`: 9 binary variants registered for ascend950:
- BF16 → BF16
- FLOAT16 → FLOAT16
- FLOAT32 → FLOAT32
- INT8 → INT8
- INT16 → INT16
- INT32 → INT32
- INT64 → INT64
- COMPLEX64 → FLOAT32
- COMPLEX32 → FLOAT16

All use ND format with `-2` (dynamic) shape and `FormatAgnostic` format matching.

Simplified key mode: `default=0` (from `abs_simplified_key.ini`).

## Test Coverage

- Unit tests: `tests/ut/op_api/test_aclnn_abs.cpp`, `tests/ut/op_host/test_abs_infershape.cpp`, `tests/ut/op_host/arch35/test_abs_tiling_arch35.cpp`
- ST tests: CSV-based parameterized tests for aclnn and kernel interfaces
- Golden data: `tests/assets/golden.py` — NumPy-based reference implementation

## CANN-to-Triton-Ascend Mapping Summary

| CANN Concept | Triton-Ascend Abstraction | Notes |
|---|---|---|
| `TPipe` / `ElementwiseSch` | `tl.load` / compute / `tl.store` | Pipeline decomposed into explicit steps |
| `Vec::CopyIn` / `Vec::CopyOut` | `tl.load` / `tl.store` | Contiguous load/store |
| `Vec::Cast` | Implicit via dtype promotion | FP32 intermediate for BF16 |
| `Vec::Abs` | `tl.abs` or `tl.where(x < 0, -x, x)` | Direct mapping |
| `EleBaseTilingData.blockNum` | `grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE` | CANN blockNum = Triton grid size |
| AIV-only | Single program on AIV | Triton Ascend backend maps programs to AIV |
| UB (Unified Buffer) | On-chip SRAM | Managed by Triton compiler |
| Workspace 16MiB | Not needed | Triton handles temporary buffers internally |
| Non-contiguous → Contiguous → compute → ViewCopy | Pointer arithmetic with strides | Triton `tl.load` with `mask` and `other` |
| Complex interleaved | Manual de-interleave | Triton loads pairs, computes `sqrt(real^2 + imag^2)` |
| DAG-based pipeline | Sequential Triton operations | CANN's DAGSch → linear Triton kernel |