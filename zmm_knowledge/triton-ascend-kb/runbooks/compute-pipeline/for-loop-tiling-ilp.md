---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.2"
title: for-loop tiling 启用双缓冲/多缓冲流水线
description: 将单次大 tile 拆分为循环小 tile，让编译器在相邻迭代间重叠 MTE2 load、Vector compute、MTE3 store，提升 PipeUtilization。
tags: [for_loop_tiling, ilp, multi_buffer, double_buffer, pipeline, num_stages]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-20 for-loop tiling 启用双缓冲/多缓冲流水线

【优化维度】：计算｜流水掩盖
【泛化层级】：条件性（迁移条件：total_elems > BLOCK_SIZE，tile 可放入 UB，内存带宽受限；失效边界：单迭代、计算极重、或复杂控制流阻止预取）

<原则：把一次大 load+compute+store 拆成循环小 tile，编译器可在迭代 i 计算的同时预取迭代 i+1 的数据、store 迭代 i-1 的结果，让 MTE2、Vector、MTE3 三阶段并行。典型收益：elementwise/normalization 15–30%。>

**无流水 vs 有流水**：

```python
# 无流水：顺序 load → compute → store
x = tl.load(ptr + offs, mask=mask)
y = compute(x)
tl.store(out_ptr + offs, y, mask=mask)

# 有流水：编译器可重叠相邻迭代
num_loop = tl.cdiv(total_elems, BLOCK_SIZE)
for i in range(num_loop):
    offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(ptr + offs, mask=offs < n)
    y = compute(x)
    tl.store(out_ptr + offs, y, mask=offs < n)
```

**`tl.static_range` vs `range`**：

| 循环类型 | 特点 | 使用场景 |
|---|---|---|
| `tl.static_range(N)` | 编译期展开，支持常量传播和多缓冲 | 迭代次数固定且编译期已知 |
| `range(tl.cdiv(N, BLOCK))` | 运行时循环，处理变长 | N 是运行时变量 |

**与 `care_padding=False` 组合**：

```python
for i in range(num_loop):
    ptr_offset = block_start + i * BLOCK_SIZE + offs
    mask = ptr_offset < n_elements
    x = tl.load(X_ptr + ptr_offset, mask=mask, other=0.0, care_padding=False)
    y = tl.exp(x)
    tl.store(Y_ptr + ptr_offset, y, mask=mask)
```

`care_padding=False` 消除 MTE2→Vector 同步屏障，与 for-loop tiling 共同实现三阶段并行。

**适用条件**：

- `total_elems > BLOCK_SIZE`（需要多迭代）
- 每个 tile 舒适地放入 UB：`BLOCK_SIZE * dtype_bytes ≤ 192KB`
- 操作是内存带宽受限（非 compute-bound）
- 循环体控制流简单，允许编译器预取

**不适用**：单迭代 kernel、重计算 kernel、复杂控制流 kernel。

**坏实践（反例）**：
```python
# 大 tensor 不分块，单 tile 超 UB
x = tl.load(ptr + tl.arange(0, 1000000))  # UB overflow
```
后果：编译失败或运行时崩溃。✅ 正解即 for-loop tiling。

## 已知实例（2026-09-21 append）

- **SelfAttention 投影/输出 GEMM（mha，ascend910b，2026-09-21）** — K 归约迭代数 = 独立杠杆：
  D=4096 投影 GEMM 的 K 循环（K=4096）主导，half BK 64→128、fp32 单独较小 BK 32→64
  （acc 128x128 fp32=64KB UB 紧），K 归约迭代数减半，speedup_vs_torch 0.1193→0.1246→0.1250。
  泛化：条件性——大 K 且 half/fp32 分档；小 K 无收益。证据：
  `output/opt_iter_1/optimized_perf_result.json.speedup_vs_torch=0.1246`、
  `output/opt_iter_2/optimized_perf_result.json.speedup_vs_torch=0.1250`。

<!-- okf:related:start -->

# 相关

- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 主题相关：共享 num_stages, pipeline
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../autotuning/triton-ascend-microkernel-autotune-noise.md) — 主题相关：共享 num_stages
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 同属优化知识，主题相关
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — 主题相关：共享 pipeline
- 相关主题: [care_padding=False 消除 MTE2→Vector 同步屏障](care-padding-false-ilp.md) — 主题相关：共享 ilp, pipeline

<!-- okf:related:end -->
