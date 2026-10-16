---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: "3.1"
title: tl.floor(x).to(int32) 直接喂 gather 地址链在边界点取址偏移一整行
description: Triton-Ascend 后端对 tl.floor→int32 结果直接参与 gather 地址计算的链，在 x 恰为 fp32 整数的边界点取址偏差一行；进入地址计算的 floor 一律改手工 floor（截断+负向修正，纯 int32 物化）
tags: [tl_floor, gather_address, backend_hazard, boundary_off_by_one, int32_materialize]
confidence: verified
status: draft
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

## tl.floor→gather 地址链边界取址 hazard（来源：grid_sample P3 iter_0→iter_1，2026-09-17）

【优化维度】：移植｜后端行为陷阱
【泛化层级】：条件性（迁移条件：floor 结果直接参与 gather/离散 load 地址计算；失效边界：floor 结果仅用于数值比较/非地址用途时无此问题；后端修复后复测）

<原则：`tl.floor(x).to(tl.int32)` 的结果**直接**喂 gather 地址链时，后端在 x 恰为 fp32 整数的边界点取址偏移一整行（行探针实测：cy=256 取到 255 行；VARIANT 对照实验证明手工 floor 无此问题）。所有进入地址计算的 floor 一律改手工 floor——向零截断 + 负向修正，纯 int32 物化：
xi = x.to(tl.int32)
floored = tl.where(x < xi.to(tl.float32), xi - 1, xi)
>

**坏实践（反例）**：
```python
x0 = tl.floor(xf).to(tl.int32)      # 边界点（xf 恰为整数）取址偏一行
corner = y0 * W_in + x0
v = tl.load(x_ptr + corner * C + cc)  # 静默错值，无报错
```
后果：fp32 用例精度偏差（实测 diff 1.8），fp16 边界点静默错行。✅ 正解：手工 floor（截断+负向修正）。

**验证**：grid_sample fp32 用例 45 修复前后 diff 1.8 → 0（出处：工作目录 output/iter_1/log.md、output/iter_1/generated_code.py `_floor_i32`）。

**已知实例**：grid_sample（gather_scatter，ascend910b/Ascend910_9372，CANN 9.1.0）

<!-- okf:related:start -->

# 相关

- 相关主题: [kernel 内索引必须转 int32，避免 int64 降级](../compute-pipeline/int32-index-cast.md) — 同属索引/地址计算纪律
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱](triton-ascend-mask-codegen-traps.md) — 同属后端行为陷阱

<!-- okf:related:end -->
