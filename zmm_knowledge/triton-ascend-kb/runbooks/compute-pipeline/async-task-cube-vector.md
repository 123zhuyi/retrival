---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.2"
title: dl.async_task 实现 Cube/Vector 异构并行
description: 在同时包含 GEMM（Cube）和 elementwise（Vector）的融合 kernel 中，用 dl.async_task 划分 scope，让两个单元并发执行。
tags: [async_task, cube, vector, heterogeneous, fusion, pipeline]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-21 dl.async_task 实现 Cube/Vector 异构并行

【优化维度】：计算｜流水掩盖
【泛化层级】：条件性（迁移条件：kernel 内同时存在 Cube 类计算和 Vector 类计算；失效边界：纯 Cube 或纯 Vector kernel，或两者之间存在强数据依赖无法重叠）

<原则：Ascend AI Core 有独立的 Cube 单元（矩阵乘）和 Vector 单元（逐元素、归约、激活）。默认情况下两者串行执行。`dl.async_task(scope="cube"|"vector")` 可把两类运算分到不同任务，让 Cube 计算 tile i+1 的同时 Vector 处理 tile i，实现异构并行。>

**正例**：

```python
import triton_ascend.language as dl

@triton.jit
def fused_gemm_norm(...):
    with dl.async_task(scope="cube"):
        acc = tl.dot(a_tile, b_tile)

    with dl.async_task(scope="vector"):
        rms = tl.sqrt(tl.sum(acc * acc, axis=1) / N + eps)
        out = (acc / rms[:, None]) * weight
        tl.store(Y_ptr + ..., out)
```

**流水线效果**：

```
Iteration i:   [Cube: GEMM tile i] ──────────────────────→
                                   [Vector: activate tile i-1] →
Iteration i+1: [Cube: GEMM tile i+1] ─────────────────────→
                                      [Vector: activate tile i] →
```

**scope 定义**：

| scope | 单元 | 典型操作 |
|---|---|---|
| `"cube"` | Cube | `tl.dot`、GEMM、batch matmul |
| `"vector"` | Vector | elementwise、`tl.exp`、normalization、reduction |

**与 persistent kernel 组合**：

```python
for row_block in range(pid, n_row_blocks, num_programs):
    with dl.async_task(scope="cube"):
        acc = tl.dot(a_tile, b_tile)

    with dl.async_task(scope="vector"):
        rms = tl.sqrt(tl.sum(acc * acc, axis=1) / n_cols + eps)
        result = acc / rms[:, None] * weight
        tl.store(Y_ptr + ..., result)
```

**注意事项**：

- `dl.async_task` 内的代码段之间若存在数据依赖，编译器会自动插入同步；需确认依赖关系正确。
- 并非所有融合 kernel 都适合，若 Vector 必须等待 Cube 全部完成则无重叠收益。
- 可配合 `dl.compile_hint("multibuffer=true")` 和 `set_workspace_multibuffer N` 使用。

**坏实践（反例）**：
```python
acc = tl.dot(a, b)        # Cube
rms = tl.sqrt(tl.sum(...)) # Vector 等待 Cube 完成
out = acc / rms
```
后果：Cube 和 Vector 串行，无法重叠。✅ 正解即用 `dl.async_task` 显式划分并发 scope。

<!-- okf:related:start -->

# 相关

- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 主题相关：共享 pipeline
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](../portability/exp_fused_matmul_rms_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [care_padding=False 消除 MTE2→Vector 同步屏障](care-padding-false-ilp.md) — 主题相关：共享 pipeline, vector
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 同属优化知识，主题相关
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](for-loop-tiling-ilp.md) — 主题相关：共享 pipeline
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 async_task, fusion

<!-- okf:related:end -->
