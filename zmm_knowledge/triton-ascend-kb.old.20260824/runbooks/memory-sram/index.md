---
kind: index
type: section_index
title: runbooks/memory-sram/（占位）
description: 分类1 内存与切块优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类1 内存与切块

覆盖二级维度：1.1 访存效率 · 1.2 Tiling/SRAM 预算 · 1.3 ND↔NZ · 1.4 32B/512B 对齐。

## 新增条目（P2 pilot + 第1批 + 剩余）

- [Cube-Vector 场景行宽必须 512B 对齐](row-width-512b-align.md) — CT-10，matmul 与 CV 向量化 load 的 512B 行宽硬约束
- [Tiling 预算与布局驱动的块大小选择](tiling-budget-and-layout-driven-selection.md) — OPT-12，UB/对齐/布局三重约束下的 BLOCK 选择
- [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — CT-11，保留硬件 block transfer 避免 2–3x 性能倒退
- [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — OPT-22，小窗口序列处理寄存器复用
- [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — OPT-23，寄存器子块提取替代重复 load
- [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — CT-14，T 维度切分保留 W-1 重叠区

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P3/P5 提取）

- [activation Memory Access 实践聚合](traj-activation-memory_access.md) — 2 条 activation 访存优化 experience 聚合
- [embedding Memory Access 实践聚合](traj-embedding-memory_access.md) — 2 条 embedding 访存优化 experience 聚合
- [gather_scatter Memory Access 实践聚合](traj-gather_scatter-memory_access.md) — 10 条 gather_scatter 访存优化 experience 聚合
- [general_ascend Memory Access 实践聚合](traj-general_ascend-memory_access.md) — 1 条 general_ascend 访存优化 experience 聚合
- [general_ascend Memory Init 实践聚合](traj-general_ascend-memory_init.md) — 1 条 general_ascend 内存初始化 experience 聚合
- [general_perf Memory Access 实践聚合](traj-general_perf-memory_access.md) — 7 条 general_perf 访存优化 experience 聚合
- [general_perf Memory Init 实践聚合](traj-general_perf-memory_init.md) — 1 条 general_perf 内存初始化 experience 聚合
- [matmul Memory Access 实践聚合](traj-matmul-memory_access.md) — 2 条 matmul 访存优化 experience 聚合
- [misc Memory Access 实践聚合](traj-misc-memory_access.md) — 13 条 misc 访存优化 experience 聚合
- [reduction Memory Access 实践聚合](traj-reduction-memory_access.md) — 1 条 reduction 访存优化 experience 聚合
- [rope Memory Access 实践聚合](traj-rope-memory_access.md) — 2 条 rope 访存优化 experience 聚合
- [softmax Memory Access 实践聚合](traj-softmax-memory_access.md) — 2 条 softmax 访存优化 experience 聚合
- [solve_tril Memory Access 实践聚合](traj-solve_tril-memory_access.md) — 1 条 solve_tril 访存优化 experience 聚合
- [ssm Memory Access 实践聚合](traj-ssm-memory_access.md) — 2 条 ssm 访存优化 experience 聚合
