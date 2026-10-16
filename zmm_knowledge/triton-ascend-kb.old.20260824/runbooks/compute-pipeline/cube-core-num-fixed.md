---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）
description: Cube matmul 类算子 grid=(CUBE_CORE_NUM,)，kernel 内交错循环处理多块，禁止 grid=(NUM_BLOCKS,)
tags: [cube_core, fixed_grid, interleaved_loop, cube_matmul]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-9 固定 CUBE_CORE_NUM 启动

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：Cube matmul 类算子；失效边界：Vector 类算子用 VEC_CORE_NUM，见 [[dynamic-vector-core-num]]）

<原则：Cube 类算子（matmul）用 CUBE_CORE_NUM（非 VEC_CORE_NUM），grid=(CUBE_CORE_NUM,) 固定启动，kernel 内 for block_idx in range(pid, NUM_BLOCKS, num_cores) 交错处理多块。禁止硬编码核数；禁止 grid=(NUM_BLOCKS,) 启动所有块。是 [[dynamic-vector-core-num]] 的 Cube 对应版。>

<骨架（占位名伪码）：
CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get('cube_core_num', 20)
grid = (CUBE_CORE_NUM,)
kernel[grid](..., num_cores=CUBE_CORE_NUM)
# kernel 内
for block_idx in range(pid, NUM_BLOCKS, num_cores):
    ...
>

**坏实践（反例）**：grid=(NUM_BLOCKS,) 或硬编码
```python
grid = (NUM_BLOCKS,)   # NUM_BLOCKS 可能远超 Cube 核数
# 或 grid = (20,)  硬编码
```
后果：超核数空跑或硬编码不适配不同芯片。✅ 正解即固定 CUBE_CORE_NUM + 交错循环。

**已知实例**：weight-quant-matmul W5

<!-- okf:related:start -->

# 相关

- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 同属优化知识，主题相关
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
