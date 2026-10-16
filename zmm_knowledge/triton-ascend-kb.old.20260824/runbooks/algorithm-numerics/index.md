---
kind: index
type: section_index
title: runbooks/algorithm-numerics/（占位）
description: 分类3 算法与数值优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类3 算法与数值（占位）

覆盖二级维度：3.1 算法重构 · 3.2 混合精度 · 3.3 数值稳定 · 3.4 跨 Block 归约。

## 新增条目（P2 第1批 + 剩余）

- [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — CT-12，fp16 exp 溢出防护与 fp32 累加
- [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — OPT-24，O(S) 内存的 online softmax

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P3/P5 提取）

- [attention Memory Access 实践聚合](traj-attention-memory_access.md) — 22 条 attention 访存优化 experience 聚合
- [attention Compute 实践聚合](traj-attention-compute.md) — 11 条 attention 计算优化 experience 聚合
- [attention Memory Numerical 实践聚合](traj-attention-memory_numerical.md) — 1 条 attention 内存+数值联合优化 experience 聚合
- [normalization Memory Access 实践聚合](traj-normalization-memory_access.md) — 24 条 normalization 访存优化 experience 聚合
- [normalization Compute 实践聚合](traj-normalization-compute.md) — 7 条 normalization 计算优化 experience 聚合
- [general_stability Numerical Stability 实践聚合](traj-general_stability-numerical_stability.md) — 2 条数值稳定性 experience 聚合
- [general_stability Numerical Precision 实践聚合](traj-general_stability-numerical_precision.md) — 2 条数值精度 experience 聚合
