---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.2"
title: care_padding=False 消除 MTE2→Vector 同步屏障
description: 当 mask 仅用于 store 过滤、load 的越界值不影响计算正确性时，tl.load 设置 care_padding=False 消除 MTE2 与 Vector 单元间的同步屏障，使流水线重叠。
tags: [care_padding, ilp, pipeline, mte2, vector, sync_barrier]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-14 care_padding=False 消除 MTE2→Vector 同步屏障

【优化维度】：计算｜流水掩盖
【泛化层级】：条件性（迁移条件：带 mask 的 load/store，且越界值在后续 store 被过滤；失效边界：越界值参与归约/累加等影响结果的操作）

<原则：Ascend AI Core 的 MTE2（load）、Vector（compute）、MTE3（store）是独立流水线。默认 `care_padding=True` 时，`tl.load` 带 mask 会在 MTE2 与 Vector 之间插入同步屏障，确保越界元素被清零后才能计算。若越界值最终会被 store 的 mask 丢弃，这个同步就是不必要的，设置 `care_padding=False` 可消除屏障，让 load 与 compute 重叠。>

**默认 vs 优化流水线**：

```
care_padding=True（有屏障）：
MTE2 load tile i → [sync] → Vector compute tile i → [sync] → MTE3 store tile i
                    ↑ 空闲等待

care_padding=False（无屏障）：
MTE2 load tile i+1 ───────────→
                    Vector compute tile i ───────→
                                      MTE3 store tile i-1 →
```

**正例**：

```python
# 默认（有同步屏障）
x = tl.load(ptr + offs, mask=mask, other=0.0)

# 优化（无同步屏障）
x = tl.load(ptr + offs, mask=mask, other=0.0, care_padding=False)

# 后续 compute
y = tl.exp(x)

# store 处仍用 mask 过滤越界值
tl.store(out_ptr + offs, y, mask=mask)
```

**安全条件**：

| 场景 | 是否安全 |
|---|---|
| elementwise：`load → compute → store(mask)` | ✅ 安全，越界值被 store mask 丢弃 |
| 归约：`tl.sum(x)`，x 含越界 padding | ❌ 不安全，padding 会污染结果 |
| 中间值参与多个下游运算 | ❌ 需逐条确认是否被 mask 过滤 |
| 明确 other=0.0 且后续是逐元素乘法/加法 | ✅ 通常安全 |

**坏实践（反例）**：
```python
x = tl.load(ptr + offs, mask=mask, care_padding=False)  # 越界值非零
s = tl.sum(x)  # 越界值被累加，结果错误
```
后果：数值错误，verify 失败。✅ 正解即 `care_padding=False` 只用于 store 已过滤的场景。

**与 for-loop tiling 的组合**：

```python
num_loop = tl.cdiv(total_elems, BLOCK_SIZE)
for i in range(num_loop):
    ptr_offset = block_start + i * BLOCK_SIZE + offs
    mask = ptr_offset < n_elements
    x = tl.load(X_ptr + ptr_offset, mask=mask, other=0.0, care_padding=False)
    y = tl.exp(x)
    tl.store(Y_ptr + ptr_offset, y, mask=mask)
```

配合 `care_padding=False`，for-loop tiling 能让 MTE2、Vector、MTE3 三阶段真正并行。

**诊断清单**：
1. profiling 中 `PipeUtilization` 低、`mte2_stall` 或 `vector_stall` 高。
2. 检查 kernel 内是否有带 mask 的 `tl.load`。
3. 确认越界值是否只被 store mask 过滤，不参与归约/累加。
4. 设置 `care_padding=False`。
5. 重跑精度和性能测试验证。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 mte2
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 mte2
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 主题相关：共享 pipeline
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — 主题相关：共享 pipeline, vector
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 同属优化知识，主题相关
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](for-loop-tiling-ilp.md) — 主题相关：共享 ilp, pipeline

<!-- okf:related:end -->
