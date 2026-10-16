---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: grid_size 必须与 num_cores 入参严格相等
description: kernel 内依赖 num_cores 入参计算循环 stride 时，grid_size 必须 == num_cores，否则任务划分错乱
tags: [grid_size, num_cores, strict_match, stride]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-5 grid_size 与 num_cores 严格匹配

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：kernel 内用 num_cores 入参算 rows_per_core=cdiv(batch,num_cores) stride；失效边界：kernel 不依赖 num_cores 入参时只需 [[grid-cap-to-core-num]] 的 grid ≤ num_cores）

<原则：当 kernel 内 rows_per_core 依赖 num_cores 入参时，grid_size 必须 == num_cores，否则部分 core 处理范围错误。是 [[grid-cap-to-core-num]] 的强化版：G4 仅要求不超核数，本条要求严格相等。>

<骨架（占位名伪码）：
grid_size = min(total_blocks, VEC_CORE_NUM)
kernel[grid_size](..., num_cores=grid_size)   # 入参严格等于 grid
>

**坏实践（反例）**：grid_size ≠ num_cores 入参
```python
grid = (min(total, VEC_CORE_NUM),)
kernel[grid](..., num_cores=VEC_CORE_NUM)  # num_cores 可能 > grid，rows_per_core 算错
```
后果：rows_per_core 公式与实际 grid 不一致，部分 core 处理范围错误（MoeGatingTopKSoftmax 实测结果错误）。✅ 正解即 num_cores=grid_size。

**已知实例**：transformer-inference T1（MoeGatingTopKSoftmax、AttentionSoftmax）

<!-- okf:related:start -->

# 相关

- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 stride
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — stride/grid 配置错误即 Part1/Part3 排查项
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 主题相关：共享 num_cores
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 主题相关：共享 num_cores

<!-- okf:related:end -->
