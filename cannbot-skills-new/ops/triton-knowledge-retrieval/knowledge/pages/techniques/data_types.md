---
title: Data Type Optimization to Avoid Scalar Fallback
category: techniques
tags: [data-types, scalar-fallback, int64, fp32, casting, fp8, mixed-precision]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# Data Type Optimization to Avoid Scalar Fallback

## Overview

Ascend Vector units support a specific set of data types for each operation. Using an unsupported type causes the compiler to fall back from Vector execution to Scalar execution — a significant performance degradation. This page covers which types cause scalar fallback, how to detect it, and how to fix it.

## Vector Operations with Unsupported Types

The following operations degrade to full scalar execution when used with unsupported types:

| Operation | Unsupported Types | Impact |
|-----------|------------------|--------|
| Vector ADD | int64 | Full scalar fallback |
| Vector CMP (comparison) | int64, int32 | Full scalar fallback |
| Vector MUL | int64 | Full scalar fallback |
| Vector DIV | int64, int32 | Full scalar fallback |

**Full scalar fallback** means the operation executes on the scalar unit, which is 32-128x slower than the Vector unit for the same operation on the same data volume.

Note: `tl.arange(0, N)` produces `int64` by default on Ascend. This means any mask computation derived directly from `tl.arange` will use int64 comparison — a common source of scalar fallback.

## Symptom: `aiv_scalar_ratio` Spike

In profiling output (`op_summary_*.csv`), `aiv_scalar_ratio` measures the fraction of execution time spent on the scalar unit.

| aiv_scalar_ratio | Interpretation |
|-----------------|----------------|
| < 5% | Normal; acceptable scalar overhead |
| 5–10% | Moderate; investigate if performance matters |
| > 10% | Significant scalar fallback; must fix |
| > 30% | Severe; major performance loss, urgent fix needed |

A sudden spike in `aiv_scalar_ratio` when changing data shapes (e.g., larger N causing int64 index overflow) is a strong indicator of this issue.

## Fix: Cast to fp32 Before Comparison or Arithmetic

The universal fix is to cast int64 (or int32) values to fp32 before using them in Vector operations:

```python
cols = tl.arange(0, BLOCK_N)        # int64 by default
cols_fp = cols.to(tl.float32)       # convert for Vector CMP
mask = cols_fp < N                  # now uses Vector CMP unit (fp32 compare)
```

## Common Pattern: Mask Generation

This is the most frequently encountered scalar fallback pattern:

```python
# WRONG: int64 comparison → scalar fallback
cols = tl.arange(0, BLOCK_N)   # dtype: int64
mask = cols < N                 # Vector CMP with int64 → SCALAR

# CORRECT: cast to fp32 first
cols = tl.arange(0, BLOCK_N)
cols_fp = cols.to(tl.float32)  # dtype: float32
mask = cols_fp < N              # Vector CMP with fp32 → VECTOR
```

**Important**: The mask itself (`bool`) is fine. Only the comparison operands need to be fp32.

Alternative: cast N to fp32 instead of the arange:

```python
cols = tl.arange(0, BLOCK_N)
N_fp = tl.full([1], N, dtype=tl.float32)
mask = cols.to(tl.float32) < N_fp
```

## FP32 Accumulation for Mixed Precision

When inputs are fp16 or bf16, always accumulate intermediate results in fp32 to avoid numerical instability and to use the Vector unit's full accumulation precision:

```python
# Input in fp16
a = tl.load(A_ptr, ...).to(tl.float32)    # cast at load
b = tl.load(B_ptr, ...).to(tl.float32)    # cast at load

# Accumulate in fp32
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc += tl.dot(a, b)                        # fp32 accumulation

# Store in fp16
tl.store(C_ptr, acc.to(tl.float16), ...)  # cast at store
```

This follows the same principle as cuBLAS and cuDNN: compute in higher precision, store in lower precision.

## FP8 Support: UB Overflow Slicing

FP8 (float8_e4m3fn, float8_e5m2) is supported on 910b2 for attention operations, but requires special handling due to limited UB capacity:

- FP8 data is 1 byte; UB holds 192K FP8 values vs 96K FP16 values
- For attention with FP8 Q/K/V, the full K/V sequence may not fit in UB
- Solution: **UB overflow slicing** — tile along the sequence dimension

```python
# FP8 attention with UB slicing
for seq_block in range(tl.cdiv(seq_len, SEQ_BLOCK)):
    k_block = tl.load(K_ptr + seq_block * SEQ_BLOCK * d_head, ...)
    # k_block is fp8; cast to fp16 for computation
    k_block_fp16 = k_block.to(tl.float16)
    scores = tl.dot(q, k_block_fp16)
    # ...
```

## Type Casting Overhead Minimization

Type casting has a small but non-zero cost. Follow these rules to minimize overhead:

1. **Cast once at load**: Cast from storage type (fp16/bf16/int8) to compute type (fp32) immediately after loading.
2. **Cast once at store**: Cast from compute type back to storage type immediately before storing.
3. **Never cast inside a loop**: Avoid repeated casts within a for-loop body. Cast the pointer offset or pre-compute the cast outside the loop.
4. **Avoid redundant round-trips**: Don't cast fp32 → fp16 → fp32 within the same kernel.

```python
# Good: cast at boundaries
for i in range(num_loop):
    x_fp16 = tl.load(X_ptr + i * BLOCK_SIZE + offs, mask=mask)
    x_fp32 = x_fp16.to(tl.float32)   # once at load
    y_fp32 = compute(x_fp32)
    tl.store(Y_ptr + ..., y_fp32.to(tl.float16), mask=mask)  # once at store
```

## int32 比较标量降级（Ascend 特有）

除 int64 外，**int32 的 LT/GT/LE/GE 比较在 Ascend 910B 上同样会标量降级**。这是 `aiv_scalar_ratio` 高的常见根因，尤其容易出现在 block_ptr `boundary_check` 生成的 mask 和手动 `tl.arange` mask 中。

```python
# 降级：i32 LT → scalar
o_k = tl.arange(0, BK)
m_k = o_k < K                       # i32 LT → scalar fallback

# 优化：转 fp32 后比较 → Vector CMP
m_k = o_k.to(tl.float32) < K.to(tl.float32)

# 2D mask 同样处理
o_i = tl.arange(0, BC)
m_i_diag = (o_i[:, None].to(tl.float32) >= o_i[None, :].to(tl.float32))
```

**关键区别**：
- `int64`：ADD/CMP/MUL 全部标量降级
- `int32`：只有 LT/GT/LE/GE 标量降级；EQ/NE 仍走向量
- `fp16/fp32`：所有比较都走向量

**诊断**：如果 profiling 中 `aiv_scalar_ratio` 高且代码中有大量 `<`、`>`、`<=`、`>=` 比较，优先转 fp32。

经验：`exp_perf_012`、`exp_ascend_int_compare_scalar_001`

## block_ptr boundary_check 中的隐式 int32 比较

`tl.make_block_ptr(..., boundary_check=(0, 1))` 在 Ascend 后端会生成逐元素的 int32 边界比较，当 tile 较小时（如 BC=16）这一开销会被放大。

```python
# 高标量开销
p = tl.make_block_ptr(base, (T, K), (H*K, 1),
                      (i_ti, i_k*BK), (BC, BK), (1, 0))
x = tl.load(p, boundary_check=(0, 1))   # 每个元素都触发 i32 比较

# 优化：手动指针 + fp32 mask
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_ti = (i_ti + o_i).to(tl.float32) < T_local.to(tl.float32)
m_k  = o_k.to(tl.float32) < K.to(tl.float32)
mask = m_ti[:, None] & m_k[None, :]
p = base + (i_ti + o_i[:, None]) * H * K + o_k[None, :]
x = tl.load(p, mask=mask, other=0.0)
```

经验：`exp_perf_011`、`exp_ascend_blockptr_boundary_scalar_001`

## Quick Reference: Type Compatibility Matrix

| Op \ Type | int8 | int16 | int32 | int64 | fp16 | bf16 | fp32 |
|-----------|------|-------|-------|-------|------|------|------|
| ADD | V | V | V | S | V | V | V |
| CMP | V | V | S | S | V | V | V |
| MUL | V | V | V | S | V | V | V |
| EXP | N/A | N/A | N/A | N/A | V | V | V |
| SQRT | N/A | N/A | N/A | N/A | V | V | V |
| DOT | V | N/A | N/A | N/A | V | V | V |

V = Vector unit, S = Scalar fallback, N/A = Not applicable

## 相关页面
- [workflows/profiling](../workflows/profiling.md) — How to measure aiv_scalar_ratio
- [api/triton_api_cheatsheet](../api/triton_api_cheatsheet.md) — Type casting API reference
- [techniques/tiling](tiling.md) — BLOCK_SIZE affects index type range
- [techniques/memory_access_optimization](memory_access_optimization.md) — block_ptr vs manual pointer 的标量/带宽权衡
- [techniques/compute_and_branch](compute_and_branch.md) — min()/tl.cdiv 标量消除
- [ops/matmul](../ops/matmul.md) — Mixed precision matmul accumulation pattern
- [ops/ssm](../ops/ssm.md) — KDA/linear attention 反向标量消除实例

## 被以下页面引用
- [ops/elementwise](../ops/elementwise.md) — int64 mask 生成的 scalar fallback 修复
- [ops/normalization](../ops/normalization.md) — fp32 混精度累加
- [ops/reduction](../ops/reduction.md) — fp32 累加器防溢出
- [ops/ssm](../ops/ssm.md) — KDA 反向 kernel 的 int32→fp32 比较优化
- [techniques/online_algorithms](online_algorithms.md) — fp32 累加数值稳定
- [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) — P1/P2 性能约束
- [workflows/profiling](../workflows/profiling.md) — aiv_scalar_ratio 诊断
