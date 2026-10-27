---
title: Instruction-Level Parallelism on Ascend NPU
category: techniques
tags: [ilp, pipeline, multibuffer, care-padding, async-task, for-loop-tiling]
hardware: [910b2, 910b3]
last_updated: 2026-07-06
---

# Instruction-Level Parallelism on Ascend NPU

## Overview

Ascend AI Core has separate pipeline stages for Memory Transfer Engine 2 (MTE2/load), Vector compute, and Memory Transfer Engine 3 (MTE3/store). When these stages have data dependencies, the pipeline stalls and throughput drops. ILP techniques decouple these stages to allow overlap.

## Problem: Load/Compute/Store Data Dependency

In a naive kernel, the execution sequence is strictly sequential:

```
MTE2 (load)  → [stall, waiting for data]
Vector (compute) → [stall, waiting for store]
MTE3 (store)
```

The load must complete before compute can start. The compute must complete before store can start. No overlap occurs. This leaves the Vector and MTE3 units idle during loading, and MTE2 idle during compute and store.

## Solution 1: `care_padding=False` in `tl.load()`

By default (`care_padding=True`), `tl.load()` with a mask inserts a synchronization barrier between MTE2 and the Vector unit to ensure masked-out elements are properly zeroed. This barrier prevents pipeline overlap.

Setting `care_padding=False` tells the compiler that the caller does not require masked elements to be zero — the mask is used for correctness of the store, not the load. This eliminates the MTE2→Vector sync barrier.

> **默认值说明**：`care_padding` 默认为 `True`（即执行边界补零）。Ascend 910b 上只有当确认 padding 元素不影响计算结果时（mask 在 store 处已过滤）才能设为 `False`。

```python
# Default (with sync barrier - pipeline stall)
x = tl.load(ptr + offs, mask=mask, other=0.0)

# Optimized (no sync barrier - pipeline continues)
x = tl.load(ptr + offs, mask=mask, other=0.0, care_padding=False)
```

**When safe to use**: When out-of-bound elements are masked at store time, so the garbage values loaded for out-of-bound positions don't affect correctness.

**When NOT safe**: When the loaded values are used in reductions where out-of-bound elements would corrupt the result (e.g., `tl.sum` without masking the result).

## Solution 2: For-Loop Tiling — Enables Multi-Buffer Pipeline

Converting a single large operation into a loop of smaller tiles allows the compiler to apply double-buffering (multi-buffer): while iteration `i` is being computed, iteration `i+1` is being loaded. This fully overlaps MTE2 with Vector.

**Without for-loop (no pipeline)**:
```python
# Single tile: load, then compute, then store (sequential)
x = tl.load(ptr + offs, mask=mask)
y = compute(x)
tl.store(out_ptr + offs, y, mask=mask)
```

**With for-loop tiling (enables ILP)**:
```python
num_loop = tl.cdiv(total_elems, BLOCK_SIZE)
for i in range(num_loop):
    x = tl.load(ptr + i * BLOCK_SIZE + offs, mask=mask)
    y = compute(x)
    tl.store(out_ptr + i * BLOCK_SIZE + offs, y, mask=mask)
```

When each iteration's tile fits in UB, the compiler can:
1. Prefetch tile `i+1` from HBM while computing tile `i`
2. Store tile `i-1` while computing tile `i`

All three pipeline stages run simultaneously.

### When For-Loop Tiling Helps

- `total_elems > BLOCK_SIZE` (multiple iterations are needed)
- Each tile fits comfortably in UB (BLOCK_SIZE * dtype_bytes ≤ 192KB)
- The operation is memory-bandwidth-limited (not compute-limited)
- Typical speedup: 15-30% for elementwise and normalization ops

### When For-Loop Tiling Does NOT Help

- Single-iteration kernels (total_elems ≤ BLOCK_SIZE)
- Heavily compute-bound kernels (matmul with large tiles)
- When the loop body has complex control flow that prevents prefetch

### When Loop Unrolling Causes Regression（循环展开的退化陷阱）

对固定次数的小循环（如 W=4 卷积核）做完全展开，虽然消除了循环开销，但存在以下风险：

**退化根因**：

1. **独立标量 load 替代 block transfer**：展开后通常需要手动索引权重元素（如 `w0=tl.load(w_ptr+0)`, `w1=tl.load(w_ptr+1)`），丧失 `make_block_ptr` 的硬件 DMA 加速。这是性能倒退的首要原因。

2. **寄存器压力激增**：展开后所有中间切片（如 4 个 x_s* slice + 4 个 w* 标量）在单表达式中同时存活。原始 for 循环每次迭代只持有一个 slice（view），展开后寄存器中的中间量从 ~8K 增至 ~24K elements（fp16）。

3. **小维度不摊还固定开销**：当 D 维度较小（≤256），tile 数少（2-16），独立的 4 条 load 指令的启动开销占比巨大，循环展开带来的转置/切片节约微不足道。

**解决方案**：

| 条件 | 策略 |
|------|------|
| D ≥ 1024 | 启用展开（收益 2.3x-3.1x） |
| D < 1024 | 保留 for 循环 + block_ptr |
| 任意 D | 混合方案：block_ptr 加载 + extract_slice 切片（保留硬件加速） |

经验：`exp_causal_conv1d_update_002`、`exp_causal_conv1d_update_003`；详见 [ops/convolution.md](../ops/convolution.md) 和 [techniques/compute_and_branch.md §1.5](compute_and_branch.md)

### 滑动窗口寄存器复用（小固定窗口序列模式）

对小固定窗口（W=2-8）的序列处理 kernel，将窗口历史元素保持在命名寄存器中，每步仅加载 1 个新元素（而非 W 个），通过寄存器移位实现窗口滑动：

```python
# 初始化：加载窗口 W 个元素到寄存器
col0 = tl.load(x_ptr + 0 * stride, mask=mask)
col1 = tl.load(x_ptr + 1 * stride, mask=mask)
col2 = tl.load(x_ptr + 2 * stride, mask=mask)
col3 = tl.load(x_ptr + 3 * stride, mask=mask)

for i in range(seq_len - W):
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3  # 寄存器内计算
    tl.store(y_ptr + i * stride, acc, mask=mask)
    # 寄存器移位：丢弃最旧，其余左移，末尾加载新元素
    col0 = col1; col1 = col2; col2 = col3
    col3 = tl.load(x_ptr + (i + W) * stride, mask=mask)
```

相比每步加载全部 W 个元素，消除 (W-1)/W 的 HBM 读（W=4 时消除 75%）。

**适用条件**：
- 窗口大小 W 固定且在编译时已知（可 `tl.constexpr`）
- W 足够小（≤8），所有窗口元素可驻留寄存器
- seq_len 足够大（≥64），初始化开销可被摊还
- 窗口滑动是**严格移位**（丢弃最旧 → 所有左移 → 末尾新元素）

**不适用**：随机窗口访问（非连续滑动）；窗口元素需要逐位随机更新而非移位。

经验：`exp_causal_conv1d_prefill_002`；详见 [ops/convolution.md](../ops/convolution.md)

## Solution 3: `dl.async_task` for Cube/Vector Async Execution

For kernels that have both GEMM (Cube unit) and elementwise (Vector unit) operations, `dl.async_task` separates the two computation types into concurrent tasks:

```python
import triton_ascend.language as dl

@triton.jit
def fused_gemm_activation_kernel(...):
    # Cube task: GEMM
    with dl.async_task(scope="cube"):
        acc = tl.dot(a_tile, b_tile)

    # Vector task: activation function (runs concurrently with next Cube iteration)
    with dl.async_task(scope="vector"):
        result = tl.sigmoid(acc)
```

- `scope="cube"`: Operations run on the Cube unit (GEMM, dot products)
- `scope="vector"`: Operations run on the Vector unit (elementwise, normalization)

This enables heterogeneous parallelism: the Cube unit executes the next GEMM tile while the Vector unit applies the activation function to the current tile.

## Compiler Hints

### `dl.compile_hint("multibuffer=true")`

Explicitly instructs the compiler to apply multi-buffer (double-buffer) optimization. Use when the compiler does not automatically detect the opportunity.

```python
import triton_ascend.language as dl

@triton.jit
def my_kernel(...):
    dl.compile_hint("multibuffer=true")
    # ... rest of kernel
```

### `set_workspace_multibuffer N`

Controls the number of double-buffer slots allocated in workspace memory. Higher values allow deeper prefetching but consume more UB space.

```python
dl.compile_hint(f"set_workspace_multibuffer {N}")
```

Typical values: N=2 (double-buffer), N=4 (quad-buffer for long-latency loads).

## Combined Example: Full ILP Optimization

```python
@triton.jit
def optimized_elementwise_kernel(
    X_ptr, Y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * NUM_LOOP

    num_loop = tl.cdiv(n_elements - block_start, BLOCK_SIZE)
    offs = tl.arange(0, BLOCK_SIZE)

    for i in range(num_loop):
        ptr_offset = block_start + i * BLOCK_SIZE + offs
        mask = ptr_offset < n_elements

        # care_padding=False: eliminates MTE2->Vector sync barrier
        x = tl.load(X_ptr + ptr_offset, mask=mask, other=0.0, care_padding=False)

        y = tl.exp(x)  # compute on Vector unit

        tl.store(Y_ptr + ptr_offset, y, mask=mask)
```

## 相关页面
- [techniques/tiling](tiling.md) — For-loop tiling structure and BLOCK_SIZE selection
- [techniques/operator_fusion](operator_fusion.md) — dl.async_task for fused kernels
- [hardware/ascend_910b](../hardware/ascend_910b.md) — Pipeline architecture, UB, MTE2/MTE3
- [workflows/profiling](../workflows/profiling.md) — Detecting pipeline stalls via profiling metrics
- [api/triton_api_cheatsheet](../api/triton_api_cheatsheet.md) — tl.load care_padding, dl.async_task reference

## 被以下页面引用
- [ops/elementwise](../ops/elementwise.md) — for-loop tiling 启用向量流水线
- [ops/normalization](../ops/normalization.md) — 减少 MTE2 同步屏障
- [ops/convolution](../ops/convolution.md) — 循环展开退化分析（D 阈值控制 + 寄存器压力）；滑动窗口寄存器复用实例
- [ops/solve_tril](../ops/solve_tril.md) — constexpr 双路径展开 vs 运行时循环性能差异
- [techniques/compute_and_branch](compute_and_branch.md) — W_IS_4 constexpr 展开 + DIM_IS_MULTIPLE 分支消除；constexpr 双路径（完整 tile vs 边界 tile §1.9）
- [techniques/memory_access_optimization](memory_access_optimization.md) — block_ptr 混合方案避免独立 load 退化（§9）；寄存器驻留 + 移位代替重复加载（§10）；preload + extract_slice（§11）
- [techniques/operator_fusion](operator_fusion.md) — dl.async_task Cube/Vector 异构流水
- [techniques/persistent_kernels](persistent_kernels.md) — 流水线与 persistent 组合
- [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) — care_padding=False 约束
