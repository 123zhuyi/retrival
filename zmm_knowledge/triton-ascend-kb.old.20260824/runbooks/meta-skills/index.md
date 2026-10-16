---
kind: index
type: section_index
title: runbooks/meta-skills/（占位）
description: 分类6 Agent 元技能优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类6 Agent 元技能

覆盖二级维度：6.1 瓶颈诊断树 · 6.2 修复链 · 6.3 轨迹动作时序。

## 新增条目（P2 pilot + 第1批 + 剩余）

- [形状驱动算子决策树](shape-driven-operator-decision-tree.md) — DG-9，按计算结构把算子路由到对应参考卡与 runbook
- [Triton-Ascend kernel 调试排查清单](debugging-checklist.md) — DG-10，内存/精度/编译错误的系统化排查链
- [Profiling 驱动的瓶颈诊断与优化路由](profiling-bottleneck-checklist.md) — DG-12，按 msprof 指标路由到对应 runbook

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P5 提取）

- [attention Workflow Opt 实践聚合](traj-attention-workflow_opt.md) — 1 条 attention 流程优化 experience 聚合
- [general_api Loop Adapt 实践聚合](traj-general_api-loop_adapt.md) — 2 条 general_api 循环适配 experience 聚合
- [general_api Syntax Adapt 实践聚合](traj-general_api-syntax_adapt.md) — 1 条 general_api 语法适配 experience 聚合
- [general_api Syntax Pitfall 实践聚合](traj-general_api-syntax_pitfall.md) — 1 条 general_api 语法避坑 experience 聚合
- [general_ascend Syntax Adapt 实践聚合](traj-general_ascend-syntax_adapt.md) — 1 条 general_ascend 语法适配 experience 聚合
- [general_ascend Syntax Pitfall 实践聚合](traj-general_ascend-syntax_pitfall.md) — 1 条 general_ascend 语法避坑 experience 聚合
- [general_ascend Syntax Spec 实践聚合](traj-general_ascend-syntax_spec.md) — 4 条 general_ascend 语法规范 experience 聚合
- [general_debug Debugging 实践聚合](traj-general_debug-debugging.md) — 6 条 general_debug 调试排查 experience 聚合
- [general_perf Perf Pitfall 实践聚合](traj-general_perf-perf_pitfall.md) — 1 条 general_perf 性能避坑 experience 聚合
- [misc Compilation Opt 实践聚合](traj-misc-compilation_opt.md) — 1 条 misc 编译优化 experience 聚合
- [solve_tril Opt Decision 实践聚合](traj-solve_tril-opt_decision.md) — 1 条 solve_tril 优化决策 experience 聚合
