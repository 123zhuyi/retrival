---
kind: index
type: section_index
title: runbooks/（占位）
description: 6 大分类优化点库（OPT-*/AP-*/CT-*/DG-*），跨算子单一共享、增量合并。
---

# runbooks/（占位）

6 大分类优化点库，跨算子单一共享、增量合并（依据 `docs/triton-knowledge-design/triton ascend 知识库内容设计.md` 的 6 分类 / 18 维度）：

- [memory-sram/](memory-sram/index.md) — 分类1 内存与切块（1.1 访存效率 · 1.2 Tiling/SRAM 预算 · 1.3 ND↔NZ · 1.4 32B/512B 对齐）
- [compute-pipeline/](compute-pipeline/index.md) — 分类2 计算与流水（2.1 单元利用/Scalar 削减 · 2.2 流水掩盖 · 2.3 多核负载/coreDim · 2.4 融合）
- [algorithm-numerics/](algorithm-numerics/index.md) — 分类3 算法与数值（3.1 算法重构 · 3.2 混合精度 · 3.3 数值稳定 · 3.4 跨 Block 归约）
- [portability/](portability/index.md) — 分类4 规范与移植（4.1 动态 shape/兼容 · 4.2 编译器约束/hints · 4.3 910B/910C/310P）
- [autotuning/](autotuning/index.md) — 分类5 调优空间设计（5.1 搜索空间剪枝/num_stages/autotune key）
- [meta-skills/](meta-skills/index.md) — 分类6 Agent 元技能（6.1 瓶颈诊断树 · 6.2 修复链 · 6.3 轨迹动作时序）
