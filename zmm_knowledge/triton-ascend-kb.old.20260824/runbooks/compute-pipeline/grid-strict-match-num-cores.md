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

- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 stride
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 stride
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 主题相关：共享 stride
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 主题相关：共享 num_cores
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 主题相关：共享 num_cores
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
