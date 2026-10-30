---
id: kernel-elementwise-abs
title: Elementwise absolute value (Abs)
type: kernel
tags: [elementwise, abs, tail_mask, contiguous_access, aiv_only, bf16, complex, dtype_dispatch]
techniques: [tail_mask, contiguous_access, fp32_accumulation]
architectures: [Ascend910B1, Ascend950PR, Ascend950DT]
kernel_types: [math]
operators: [abs]
symptoms: []
confidence: high
reproducibility: source
artifact_dir: artifacts/elementwise-abs
related: [technique-tail-mask, technique-contiguous-load-order, technique-fp32-accumulation, kernel-elementwise-add, source-cann-ops-math-abs]
sources: [source-cann-ops-math-abs]
performance_claims: []
---

# Elementwise absolute value (Abs)

## Semantic

`y = abs(x)` — element-wise absolute value. Output shape equals input shape.

## Data Type Handling

### Real dtype

| Input dtype | Output dtype | Compute dtype | Intermediate |
|-------------|--------------|---------------|---------------|
| float32     | float32      | float32       | none |
| float16     | float16      | float16       | none |
| bfloat16    | bfloat16     | float32       | FP32 intermediate |
| int8/16/32/64 | same       | same          | none |
| uint8/bool  | same         | passthrough   | identity (no-op) |

### Complex dtype

| Input dtype | Output dtype | Formula |
|-------------|--------------|---------|
| COMPLEX32   | float16      | `sqrt(real^2 + imag^2)` |
| COMPLEX64   | float32      | `sqrt(real^2 + imag^2)` with numerical stability |

For COMPLEX64, the CANN kernel uses a numerically stable algorithm:
1. Sort `|real|` and `|imag|` into max/min
2. Normalize to [1, 2) range to avoid overflow
3. Compute `sqrt(norm_max^2 + norm_min^2)` and multiply back exponent
4. Handle subnormals and zero separately

## Tiling Key Dispatch

| Key | Condition | Description |
|-----|-----------|-------------|
| 101 | Output dtype == BF16 | BF16 input, FP32 intermediate, BF16 output |
| 102 | Real dtype, not BF16 | Direct same-dtype abs |
| 103 | COMPLEX32 or COMPLEX64 | Complex abs with promoted output |

## Kernel Pipeline

### Real dtype DAG

```
CopyIn → Cast → Abs → Cast → CopyOut
```

For BF16 (key 101): `CopyIn<bfloat16> → Cast<float, bfloat16, 0> → Abs<float> → Cast<bfloat16, float, 1> → CopyOut<bfloat16>`

The FP32 intermediate cast prevents precision loss in the abs operation.

For other real dtypes (key 102): Cast nodes are identity (same dtype in/out); the pipeline reduces to `CopyIn → Abs → CopyOut`.

### Complex dtype DAG

```
CopyIn → AbscomplexCustom → CopyOut
```

The custom AbscomplexCustom operation handles interleaved real/imag pairs with the numerical stability algorithm described above.

## AIV-only Execution

The kernel is marked `KERNEL_TYPE_AIV_ONLY` — it runs on vector (AIV) cores only, not on cube (AIC) cores. This is the standard dispatch for elementwise vector math.

## Triton-Ascend Skeleton

The key Triton primitives used for an elementwise abs kernel:

1. `tl.program_id(0)` — 1D grid index
2. `BLOCK_SIZE` — tile width
3. Offsets: `offsets = start + tl.arange(0, BLOCK_SIZE)`
4. Tail mask: `mask = offsets < n_elements`
5. `tl.load(x_ptr + offsets, mask=mask, other=0.0)`
6. `tl.abs(x)` or `tl.where(x < 0, -x, x)` for compute
7. `tl.store(y_ptr + offsets, result, mask=mask)`

For BF16, load as BF16, cast to FP32, compute abs, cast back to BF16, store.

For complex, assume interleaved `[real0, imag0, real1, imag1, ...]` layout — load pairs, compute `tl.sqrt(real*real + imag*imag)` manually.

## References

- Source: `source-cann-ops-math-abs` — full CANN ops-math Abs operator analysis
- Tail mask: `technique-tail-mask` — boundary predicate pattern
- Contiguous load: `technique-contiguous-load-order` — neighboring lane mapping
- FP32 accumulation: `technique-fp32-accumulation` — BF16 intermediate precision
- Elementwise pattern: `kernel-elementwise-add` — sibling elementwise kernel