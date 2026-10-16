---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK
description: 1D grid 超 65535 启动失败；BLOCK 过大触发 Vector Core timeout（507034）
tags: [grid_limit, 65535, vector_timeout, 507034, block_too_large]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-11 grid 上限与 BLOCK 超时

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：total_blocks 大或 BLOCK 过大；失效边界：total_blocks 小且 BLOCK 合理）

<原则：1D grid 硬上限 65535，超出启动失败；BLOCK 过大在单 program 内循环过多触发 Vector Core timeout（507034）。total_blocks 超 65535 时用 2D grid 或增大 BLOCK 减少 block 数；BLOCK 取 next_pow2_geq(dim) 但不超过实际维度。与 [[grid-cap-to-core-num]] 配套。>

<骨架（占位名伪码）：
total_blocks = cdiv(total_elems, BLOCK)
if total_blocks > 65535:
    # 增大 BLOCK 或用 2D grid
    BLOCK = next_pow2_geq(dim) * 2
    total_blocks = cdiv(total_elems, BLOCK)
grid = (min(total_blocks, num_cores),)
# BLOCK 不超过实际维度，避免单 program 循环过多超时
>

**坏实践（反例）**：1D grid 超 65535 或 BLOCK=256 固定
```python
grid = (total_blocks,)   # total_blocks > 65535 启动失败
# 或 BLOCK=256 固定，小 dim 时单 program 循环过多 -> 507034 timeout
```
后果：启动失败或 Vector Core timeout。✅ 正解即 2D grid/增大 BLOCK + BLOCK 适配实际维度。

**已知实例**：index-computation G11

<!-- okf:related:start -->

# 相关

- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
