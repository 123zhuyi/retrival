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

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_004` — 内存访问优化 （收益 20%）
  - 所有带多维内存访问的Triton kernel都需严格验证stride参数，优先传递tensor.stride()的原生值，避免手动计算；910b2/910b3平台对stride不连续的访存惩罚尤为严重
- `exp_perf_006` — 内存访问优化 （收益 35%）
  - 适用于所有非连续张量的元素级运算场景，910b2/910b3平台下，即使包含`contiguous()`的一次性拷贝，整体性能仍比直接stride访问高30%以上

<!-- okf:related:start -->

# 相关

- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 主题相关：共享 num_stages, pipeline
- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 num_stages
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — 主题相关：共享 pipeline
- 相关主题: [care_padding=False 消除 MTE2→Vector 同步屏障](care-padding-false-ilp.md) — 主题相关：共享 ilp, pipeline
- 相关主题: [循环不变量外提（LICM）减少重复地址计算](loop-invariant-code-motion.md) — 同属优化知识，主题相关
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
