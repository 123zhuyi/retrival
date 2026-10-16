---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.3"
title: 坐标比较必须先转 float32
description: 禁止对整数坐标直接 tl.where(coord < 0,...)，必须 .to(tl.float32) 再比较，避免整数比较降级和 cast 向零截断
tags: [coord_compare, float32, int_compare_demote, cast_truncate]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-1 坐标比较转 float32

【优化维度】：数值｜数值稳定
【泛化层级】：Triton通用（迁移条件：kernel 内对坐标做比较/分支；失效边界：纯非负整数比较无截断问题时可放宽，但仍建议统一转 fp32）

<原则：两个问题叠加--(1) Triton Ascend 整数比较可能降级；(2) tl.cast 对负数是向零截断而非 floor，坐标计算（如 src_y = scale*(i+0.5)-0.5 可能为负）若用 int 比较+cast 会导致 t 偏移错误。统一先转 float32 再比较。与 [[fp32-reduction-accumulator]] 同属数值稳定。>

<骨架（占位名伪码）：
coord_f = coord.to(tl.float32)              # 先转 fp32
y0 = tl.cast(tl.floor(coord_f), tl.int32)
# 负坐标 floor 修正
y0 = tl.where(coord_f < y0, y0 - 1, y0)     # fp32 比较，修正 cast 截断
>

**坏实践（反例）**：整数坐标直接比较
```python
coord_i = tl.cast(coord, tl.int32)          # 负数向零截断（非 floor）
result = tl.where(coord_i < 0, fallback, val)  # 整数比较可能降级
```
后果：负坐标 cast 截断导致 t 偏移错误，verify 失败（常见于上采样首行/首列）。✅ 正解即先转 float32 再比较。

**已知实例**：tensor-transform G8、Interpolate L1.11（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 主题相关：共享 float32

<!-- okf:related:end -->
