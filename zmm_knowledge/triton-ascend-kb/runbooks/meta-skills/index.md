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


## 新增条目（raw experiences 导入，待 review）

- [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](exp_bonus_renew_001.md) — 910b2/910b3/91093/91095 + int32/int64 下变长序列的标量更新 kernel，对 max_spec_len 标 do_not_specialize 避免按长度特化出多份二进制；地址用 position*(max_spec_len+1)+num_tokens 内联计算；单标量操作保持最小寄存器压力，编译时间降 >80%
- [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](exp_debug_001.md) — 910b2/910b3/910b4 上 kernel 崩溃/数据错乱先查内存访问：所有 tl.load/tl.store 带 mask 或 boundary_check、stride 参数与张量实际布局一致、所有索引进 offset < n_elements 判断
- [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](exp_debug_002.md) — 910b2/910b3/910b4 上 kernel 编译失败先查控制流：return/break/continue 须改 mask 条件执行，链式布尔/嵌套条件须分步算 mask 再组合，tl.constexpr 仅用于内核参数声明
- [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](exp_debug_003.md) — 910b2/910b3/910b4 上 kernel 启动失败先查 grid：各维乘积须 ≤65535、必须为 ≤3 维 tuple；大 shape 算子用 for i in range(pid, total, core_num) 交错循环规避超限
- [并发写入必须用原子操作且类型与目标内存严格匹配](exp_debug_004.md) — 910b2/910b3/910b4 上多核并发写同一地址必须用 tl.atomic_add/atomic_max 等原子操作，且原子操作 dtype 与目标内存 dtype 严格一致；910b 仅支持 add/max/min 基础原子操作
- [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](exp_debug_005.md) — 910b2/910b3/910b4 上切片相关编译失败先查：b[0]/b[i:j] 等 Python 切片须改 tl.get_element/tl.extract_slice/tl.insert_slice；tl.arange 编译期索引表达式禁调 get_element，直接数值计算
- [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](exp_debug_triton_branch_type_mismatch_001.md) — 910b2/910b3 + bf16/fp16 下 "Mismatched type for variable between then block and else block" 修复链：if/else 分支改 masked load + other=0.0；bool 张量 Ascend load 后为 i8 不能作 mask，host 侧转 int32 再比较；滑窗循环越界时 tl.where 窗口值被零覆盖
- [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](exp_solve_tril_006.md) — 910b2/910b3/91093/91095 + fp16/bf16/fp32 下三角求解类小矩阵（16/32/64）优化决策：1×16×16×16 用 VEC 广播乘+sum 比 tl.dot 快 14-20x；99% tile 走同路径时保留 constexpr 双分支；preload 按 grid 规模（B≥4,T≥1024 启用）；off-diagonal 按需 HBM 加载防寄存器 spill；逐行递推 O(BT²) 依赖是算法根本限制

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

## 新增条目（GDN 三篇优化实践蒸馏 + staging 合入，2026-08-27）

- [流水图驱动优化：先可视化找气泡，按气泡边类型分派手段](pipeline-diagram-driven-diagnosis.md) — 气泡五分类→手段映射表，GDN 流水图 + PyPTO 泳道图双案例
- [跨 run benchmark 噪声鉴别纪律](cross-run-benchmark-noise.md) — framework 侧 3-7x 波动，跨 round 结论必须同 run impl 侧复核
