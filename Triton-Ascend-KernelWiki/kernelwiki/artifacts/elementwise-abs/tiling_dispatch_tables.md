# Abs Operator Tiling and Dispatch Tables

## Tiling Key Dispatch

| Tiling Key | Name | Condition | Storage Dtype (U) | Compute Dtype (T) | Output Dtype | Notes |
|-----------|------|-----------|-------------------|-------------------|-------------|-------|
| 101 | `ELEMENTWISE_BF16` | output dtype == BF16 | `bfloat16_t` | `float` | `bfloat16_t` | FP32 intermediate prevents precision loss |
| 102 | `ELEMENTWISE_OTHER` | real dtype, not BF16 | `dtype` | `dtype` | `dtype` | Direct abs, Cast is identity |
| 103 | `ELEMENTWISE_COMPLEX` | COMPLEX32 or COMPLEX64 | `complex32`/`complex64` | `float`/`float` | `float16`/`float` | Numerical stability algorithm |

Source: `op_host/arch35/abs_tiling_arch35.cpp` lines 30-33, `op_kernel/abs_apt.cpp` lines 33-73.

## Dtype → Tiling Key Mapping

| Input Dtype | Tiling Key | DAG Template |
|-------------|-----------|--------------|
| BF16 | 101 | `AbsDag<bfloat16_t, float>::OpDag` |
| FLOAT16 | 102 | `AbsDag<half, half>::OpDag` |
| FLOAT | 102 | `AbsDag<float, float>::OpDag` |
| INT8 | 102 | `AbsDag<int8_t, int8_t>::OpDag` |
| INT16 | 102 | `AbsDag<int16_t, int16_t>::OpDag` |
| INT32 | 102 | `AbsDag<int32_t, int32_t>::OpDag` |
| INT64 | 102 | `AbsDag<int64_t, int64_t>::OpDag` |
| COMPLEX64 | 103 | `AbscomplexDag<complex64, float>::OpDag` |
| COMPLEX32 | 103 | `AbscomplexDag<complex32, half>::OpDag` |

Source: `op_host/arch35/abs_tiling_arch35.cpp` lines 92-109.

## Kernel Entry Dispatch

| Tiling Key | `DTYPE_X` | ElementwiseSch Template |
|-----------|-----------|------------------------|
| 101 | (compile-time) | `ElementwiseSch<0, AbsDag<bfloat16_t, float>::OpDag>` |
| 102 | `half` | `ElementwiseSch<0, AbsDag<half, half>::OpDag>` |
| 102 | `float` | `ElementwiseSch<0, AbsDag<float, float>::OpDag>` |
| 102 | `int8_t` | `ElementwiseSch<0, AbsDag<int8_t, int8_t>::OpDag>` |
| 102 | `int16_t` | `ElementwiseSch<0, AbsDag<int16_t, int16_t>::OpDag>` |
| 102 | `int32_t` | `ElementwiseSch<0, AbsDag<int32_t, int32_t>::OpDag>` |
| 102 | `int64_t` | `ElementwiseSch<0, AbsDag<int64_t, int64_t>::OpDag>` |
| 103 | `complex64` | `ElementwiseSch<0, AbscomplexDag<complex64, float>::OpDag>` |
| 103 | `complex32` | `ElementwiseSch<0, AbscomplexDag<complex32, half>::OpDag>` |

Source: `op_kernel/abs_apt.cpp` lines 33-73.

## Binary Config (ascend950)

| Binary Hash | Input Dtype | Output Dtype |
|------------|-------------|--------------|
| `Abs_1c4543fdfe7149b6b0717bd88f061562` | BF16 | BF16 |
| `Abs_d2db1a80c523e7e59a032c95969880af` | FLOAT16 | FLOAT16 |
| `Abs_5c5e58e043224cccf51a06108010ae13` | FLOAT32 | FLOAT32 |
| `Abs_6c5e58e043224edaf51a06108010ae56` | INT8 | INT8 |
| `Abs_9c5e58e998224edaf51a06202410ae12` | INT64 | INT64 |
| `Abs_8098b79498645e9c0da095babdeaf862` | INT32 | INT32 |
| `Abs_df11d48f8de6e91ff319613e8e50dc14` | COMPLEX64 | FLOAT32 |
| `Abs_df11d48f8de6e91ff319613e8e50dd12` | INT16 | INT16 |
| `Abs_df11d48f8de6e91ff319613e8e50dd98` | COMPLEX32 | FLOAT16 |

Source: `op_host/config/ascend950/abs_binary.json`.

All entries: ND format, dynamic shape (-2), FormatAgnostic.

## Host Tiling Parameters

| Parameter | Source | Value |
|-----------|--------|-------|
| coreNum | `ascendcPlatform.GetCoreNumAiv()` | AIV core count |
| ubSize | `ascendcPlatform.GetCoreMemSize(UB)` | UB size in bytes |
| workspaceSize | hardcoded | 16 MiB (`16777216`) |
| blockDim | `tiling->baseTiling.blockNum` | Determined by ElewiseBaseTiling |
| TilingPrepare | `TilingPrepareForAbs` | Pre-tiling platform query |
| Tiling | `TilingForAbs` | Main tiling entry |

Source: `op_host/arch35/abs_tiling_arch35.cpp` lines 33-148.

## Platform Support Matrix

| Platform | AICore Dtype Support | Complex Support | BF16 Support |
|----------|---------------------|-----------------|--------------|
| Ascend910 (DAV_210) | FLOAT, FLOAT16, INT32 | No | No |
| Ascend910B (DAV_2201) | + BF16, INT64 | COMPLEX64 only | Yes |
| Ascend950 (RegBase) | + INT8, INT16, COMPLEX32, COMPLEX64 | Both | Yes |
| Atlas A3 (RegBase) | Same as 950 | Both | Yes |
| Ascend610Lite (DAV_3102) | FLOAT, FLOAT16, INT32 | No | No |

Source: `op_api/abs.cpp` lines 25-52, `op_api/aclnn_abs.cpp` lines 31-43.