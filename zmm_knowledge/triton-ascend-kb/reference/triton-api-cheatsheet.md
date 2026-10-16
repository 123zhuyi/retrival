---
schema_version: okf.v1
kind: api
type: api_reference
source_family: reference
title: Triton API Quick Reference for Ascend Development
description: Triton API Quick Reference for Ascend Development
tags: [api, cheatsheet, ascend_extensions, dl, tle, autotune, 910b2, 910b3]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/api/triton_api_cheatsheet.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/api/triton_api_cheatsheet.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/api/triton_api_cheatsheet.md` 迁移而来，原始为 API 速查表，未做内容改写，仅转 OKF frontmatter。

# Triton API Quick Reference for Ascend Development

## Kernel Declaration

| API | Signature | Description |
|-----|-----------|-------------|
| `@triton.jit` | `@triton.jit` | Compile kernel for device execution |
| `@triton.autotune` | `@triton.autotune(configs, key)` | Auto-select best config from list |
| `@triton.heuristics` | `@triton.heuristics({'KEY': fn})` | Dynamic config based on input |
| `tl.constexpr` | Type annotation | Compile-time constant (enables specialization) |

**Ascend note**: `tl.constexpr` parameters are evaluated at JIT compile time, enabling dead-code elimination. Use for `HAS_BIAS`, `HAS_RESIDUAL`, etc.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 16}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

## Program Identity

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.program_id` | `tl.program_id(axis: int)` | Current program's ID along axis (0, 1, 2) |
| `tl.num_programs` | `tl.num_programs(axis: int)` | Total number of programs along axis |

```python
pid = tl.program_id(axis=0)
num_programs = tl.num_programs(axis=0)
# Grid-stride: for row in range(pid, n_rows, num_programs)
```

## Memory

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.load` | `tl.load(ptr, mask=None, other=0, cache_modifier='', care_padding=True)` | Load from HBM to register |
| `tl.store` | `tl.store(ptr, val, mask=None)` | Store from register to HBM |
| `tl.make_block_ptr` | `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | 2D block pointer |
| `tl.advance` | `tl.advance(ptr, offsets)` | Advance a block pointer |

**Ascend notes**:
- `care_padding=False`: Eliminates MTE2→Vector sync barrier; safe when out-of-bound values are masked at store. **Use this for ILP.**
- `cache_modifier='.ca'`: Cache in L2 (streaming). `'.cg'`: Cache at L1 level.
- `tl.make_block_ptr` with `order=(1, 0)` for row-major (standard for Dot API on 910B2).

```python
# Standard load with ILP optimization
x = tl.load(ptr + offs, mask=mask, other=0.0, care_padding=False)

# 2D block pointer for matmul
a_block_ptr = tl.make_block_ptr(
    base=A, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
)
a = tl.load(a_block_ptr)
a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
```

## Indexing

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.arange` | `tl.arange(start, end)` | Integer range tensor (end - start must be power of 2) |
| `tl.reshape` | `tl.reshape(x, shape)` | Reshape tensor |
| `tl.view` | `tl.view(x, shape)` | View with different shape (zero-copy) |
| `tl.trans` | `tl.trans(x)` | Transpose 2D tensor |
| `tl.permute` | `tl.permute(x, dims)` | Permute dimensions |
| `tl.expand_dims` | `tl.expand_dims(x, axis)` | Add dimension of size 1 |

**Ascend note**: `tl.arange(0, N)` produces **int64** by default. For masking, cast to fp32 first to avoid scalar fallback:
```python
cols = tl.arange(0, BLOCK_N).to(tl.float32)  # avoid int64 CMP scalar fallback
mask = cols < N
```

## Math Operations

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.exp` | `tl.exp(x)` | Element-wise exp |
| `tl.log` | `tl.log(x)` | Element-wise natural log |
| `tl.sqrt` | `tl.sqrt(x)` | Element-wise square root |
| `tl.rsqrt` | `tl.rsqrt(x)` | Element-wise reciprocal square root |
| `tl.abs` | `tl.abs(x)` | Element-wise absolute value |
| `tl.sin` | `tl.sin(x)` | Element-wise sine |
| `tl.cos` | `tl.cos(x)` | Element-wise cosine |
| `tl.sigmoid` | `tl.sigmoid(x)` | Element-wise sigmoid: 1/(1+exp(-x)) |
| `tl.tanh` | `tl.tanh(x)` | Element-wise tanh |
| `tl.erf` | `tl.erf(x)` | Element-wise error function (for GELU) |

All math ops require floating-point input (fp16, bf16, or fp32). They do NOT work with int types.

## Reduction Operations

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.sum` | `tl.sum(x, axis=None, keep_dims=False)` | Sum reduction |
| `tl.max` | `tl.max(x, axis=None, keep_dims=False)` | Max reduction |
| `tl.min` | `tl.min(x, axis=None, keep_dims=False)` | Min reduction |
| `tl.reduce` | `tl.reduce(x, axis, combine_fn)` | Custom reduction with combine function |

```python
# Row-wise sum for LayerNorm
row_sum = tl.sum(x, axis=1, keep_dims=True)   # [BLOCK_M, 1]

# Scalar max for softmax
m = tl.max(scores, axis=1)  # [BLOCK_M]
```

## Compute Operations

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.dot` | `tl.dot(a, b, acc=None, out_dtype=tl.float32)` | Matrix multiply: a @ b |
| `tl.where` | `tl.where(cond, x, y)` | Conditional select |
| `tl.cdiv` | `tl.cdiv(a, b)` | Ceiling integer division |

**Ascend note for `tl.dot`**:
- Input tensors must have rank 2
- Output is float32 accumulator by default
- `out_dtype=tl.float16` for fp16 output accumulation (less precise, faster)
- For 910B2: use `order=(1, 0)` in `tl.make_block_ptr` for compatible format

```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
a = tl.load(a_ptr)   # [BLOCK_M, BLOCK_K]
b = tl.load(b_ptr)   # [BLOCK_K, BLOCK_N]
acc += tl.dot(a, b)  # fp32 accumulation
```

## Atomic Operations

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.atomic_add` | `tl.atomic_add(ptr, val, mask=None)` | Atomic add to memory |
| `tl.atomic_max` | `tl.atomic_max(ptr, val, mask=None)` | Atomic max to memory |
| `tl.atomic_cas` | `tl.atomic_cas(ptr, cmp, val)` | Compare-and-swap |

Used for Split-K reduction patterns where multiple programs write to the same output location.

```python
# Split-K: atomically accumulate partial sums
tl.atomic_add(C_ptr + offs, partial_sum, mask=mask)
```

## Type Operations

| API | Signature | Description |
|-----|-----------|-------------|
| `x.to(dtype)` | `tensor.to(tl.float32)` | Cast tensor to dtype |
| `tl.float32` | Type constant | 32-bit float |
| `tl.float16` | Type constant | 16-bit float |
| `tl.bfloat16` | Type constant | Brain float 16 |
| `tl.int32` | Type constant | 32-bit integer |
| `tl.int64` | Type constant | 64-bit integer (avoid for CMP/ADD) |
| `tl.int8` | Type constant | 8-bit integer (for quantization) |
| `tl.float8e4m3fn` | Type constant | FP8 E4M3 format |

**Critical Ascend type rules**:
- int64 ADD → scalar fallback → cast to fp32
- int64 CMP → scalar fallback → cast to fp32
- int32 CMP → scalar fallback → cast to fp32
- Always accumulate in fp32 when inputs are fp16/bf16

## Control and Debug

| API | Signature | Description |
|-----|-----------|-------------|
| `tl.static_assert` | `tl.static_assert(cond, msg)` | Compile-time assertion |
| `tl.device_assert` | `tl.device_assert(cond, msg)` | Runtime assertion (debug only) |
| `tl.debug_barrier` | `tl.debug_barrier()` | Memory fence for debugging |

## Ascend Extensions

### `dl.async_task`

```python
import triton_ascend.language as dl

with dl.async_task(scope="cube"):
    acc = tl.dot(a, b)       # runs on Cube unit

with dl.async_task(scope="vector"):
    out = tl.sigmoid(acc)    # runs on Vector unit (concurrent with next Cube)
```

| scope | Unit | Use for |
|-------|------|---------|
| `"cube"` | Cube unit | tl.dot, GEMM |
| `"vector"` | Vector unit | elementwise, norm, reduction |

### `dl.compile_hint`

```python
dl.compile_hint("multibuffer=true")           # enable double-buffer
dl.compile_hint("dot_pad_only_k")             # Cube K-only padding
dl.compile_hint("enable_hivm_auto_cv_balance") # auto CV balance
dl.compile_hint("tile_mix_vector_loop=4")     # Vector split granularity
dl.compile_hint("tile_mix_cube_loop=2")       # Cube split granularity
dl.compile_hint("set_workspace_multibuffer 4") # 4-level prefetch
```

### `tle.index_select_simd`

```python
import triton_ascend.language.extra as tle

# SIMD-optimized index select (faster than gather for contiguous ranges)
selected = tle.index_select_simd(src, indices)
```

Use for MoE expert routing and attention with custom index patterns.

### `dl.grouped_launch_diagonal`

```python
pid_m, pid_n = dl.grouped_launch_diagonal(
    pid,       # int: linear program ID
    num_pid_m, # int: tiles in M dimension
    num_pid_n, # int: tiles in N dimension
    GROUP_SIZE # int: tiles per diagonal group (8, 16, or 32)
)
```

Hardware-level diagonal PID mapping for L2 bank conflict elimination. 910B2 only. **Do NOT use `tl.swizzle2d` instead.**

## Quick Ascend Gotcha Reference

| Gotcha | Fix |
|--------|-----|
| `tl.arange` returns int64, CMP is scalar | `.to(tl.float32)` before comparison |
| `tl.load` with mask causes pipeline stall | Add `care_padding=False` |
| Grid > num_aicores fails | Use 1D grid ≤ num_aicores, or set TRITON_ALL_BLOCKS_PARALLEL=1 |
| 2D grid shape mismatch | Use 1D grid + manual index compute |
| L2 bank conflict on large matmul | Use `dl.grouped_launch_diagonal` |
| UB overflow | Reduce BLOCK_M/BLOCK_N/BLOCK_K |
| Scalar fallback on int32 ADD | Cast to fp32 |
| `tl.swizzle2d` for diagonal | Replace with `dl.grouped_launch_diagonal` |

## 相关页面
- techniques/data_types — Scalar fallback types and fixes
- techniques/ilp — care_padding and dl.async_task details
- techniques/core_mapping — dl.grouped_launch_diagonal usage
- [hardware/ascend_910b](ascend-910b.md) — dl.compile_hint options and env vars
- workflows/profiling — Profiling APIs and metrics

<!-- okf:related:start -->

# 相关

- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](../runbooks/autotuning/autotune-debug.md) — 主题相关：共享 autotune
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../runbooks/autotuning/autotune-num-warps-stages.md) — 主题相关：共享 autotune
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../runbooks/autotuning/exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../runbooks/autotuning/triton-ascend-microkernel-autotune-noise.md) — 主题相关：共享 autotune
- 相关主题: [Scheduling Optimization: autotune block m](../runbooks/compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../runbooks/memory-sram/tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 autotune
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../runbooks/portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](../runbooks/portability/exp_sparse_flash_attn_prefill_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
