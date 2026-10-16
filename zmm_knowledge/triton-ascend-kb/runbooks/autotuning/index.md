---
kind: index
type: section_index
title: runbooks/autotuning/（占位）
description: 分类5 调优空间设计优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类5 调优空间设计（占位）

覆盖二级维度：5.1 搜索空间剪枝/num_stages/autotune key。

## 新增条目（P2 剩余）

- [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) — OPT-26，覆盖 warp/stages 组合以隐藏延迟和优化流水线


## 新增条目（raw experiences 导入，待 review）

- [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](exp_perf_001.md) — 910b2/910b3 上块大小按算子类型选型（元素级 1024、矩阵级 256 起），候选取 2 的幂次 128/256/512；可执行判据：blockSize×dtypeBytes 须对齐 32B 访存粒度、单块内存 ≤UB 预算、BLOCK_SIZE <65536，声称收益 20%
- [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](exp_perf_017.md) — 910b2/910b3 上单一 autotune 配置无法覆盖不同 shape/寄存器压力的最优调度，配置空间应含 num_warps 2/4/8 × num_stages 2/3/4 及不同 tile（BK/BV）；寄存器压力大的 kernel 须剪掉高 warp 配置防 spill，tuned 后额外加速 10-20%

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P5 提取）

- [general_ascend Tuning Config 实践聚合](traj-general_ascend-tuning_config.md) — 1 条 general_ascend 调优配置 experience 聚合
- [general_perf Tuning Config 实践聚合](traj-general_perf-tuning_config.md) — 1 条 general_perf 调优配置 experience 聚合

## 新增条目（ApplyRotaryPosEmb P4 后端契约沉淀，2026-09-09）

- [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](triton-ascend-microkernel-autotune-noise.md) — 三组一致实测：autotune do_bench 噪声误选（4.1228<4.2060）、num_stages 静默无效；小 kernel 用静态 BLOCK 规则

## 新增条目（Gdn2Fwd 回流试点，2026-09-14）

- [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](tiling-candidate-compile-cost-gate.md) — CT-1，BT=64 编译 >400s + aicore trap 实证封死调参空间
