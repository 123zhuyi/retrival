---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: 禁止在 kernel 内使用 continue/break，用 tl.where 向量分段
description: Ascend 后端对非规整控制流支持有限，continue/break 易引发编译或性能问题，必须用 tl.where 替代
tags: [no_continue, no_break, tl_where, vector_branch, control_flow]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-8 禁止 continue/break

【优化维度】：计算｜Scalar 削减
【泛化层级】：Triton通用（迁移条件：所有 kernel 内条件跳转；无失效边界）

<原则：Ascend 后端对非规整控制流支持有限，continue/break 易引发编译或性能问题。向量数据上的条件必须用 tl.where 做分段，而非标量 if 跳转。与 [[scalar-load-channel-index]] 同属 Ascend 控制流约束。>

<骨架（占位名伪码）：
# 禁止：for + continue
for i in range(n):
    if cond[i]:
        continue          # ❌ 非规整控制流
    ...

# 正确：tl.where 向量分段
mask = cond
result = tl.where(mask, fallback, val)   # ✅ 向量分段
>

**坏实践（反例）**：kernel 内用 continue/break
```python
for i in range(BLOCK_SIZE):
    if offsets[i] >= n:
        continue          # Ascend 编译/性能问题
```
后果：编译失败或性能退化。✅ 正解即 tl.where 向量分段。

**已知实例**：quantization Q7（量化算子 kernel 内统一用 tl.where）

<!-- okf:related:start -->

# 相关

- 相关主题: [Ascend 条件表达式适配：tl.where 与 if-else 的选择](../meta-skills/traj-general_api-syntax_adapt.md) — 主题相关：共享 tl_where
- 相关主题: [Triton Kernel 通用语法避坑清单](../meta-skills/traj-general_api-syntax_pitfall.md) — 主题相关：共享 control_flow
- 相关主题: [Ascend 后端避免用 tl.where 动态计算内存偏移](../meta-skills/traj-general_ascend-syntax_adapt.md) — 主题相关：共享 tl_where
- 相关主题: [通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化](traj-general_perf-scalar_elimination.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
