---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: autotuning
dimension: "5.1"
title: µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效
description: Triton-Ascend 上 num_stages/num_warps 调优运行时仅 WARNING 不生效；µs 级小 kernel 的 @triton.autotune 因 do_bench 测量噪声系统性误选配置（实测 autotune 轮 4.1228x < 静态规则轮 4.2060x）。小 kernel 用静态 BLOCK 规则，大 kernel 才考虑 autotune 且须人工复核选中项。
tags: [autotune, num_stages, num_warps, do_bench_noise, microkernel, static_block_rule, unsupported_feature]
confidence: verified
status: verified
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-09T00:00:00Z"
---

## µs 级 kernel 禁用 autotune；num_stages/num_warps 不生效（来源：ApplyRotaryPosEmb P4 三组一致实测，2026-09-08）

【优化维度】：调优｜autotune
【泛化层级】：条件性（迁移条件：单 case 延迟 µs 级的小 kernel；失效边界：大 kernel / 后续后端版本支持 num_stages 后复测）

<原则：Triton-Ascend 上 ① `num_stages`/`num_warps` 传入后运行时不生效（WARNING 级别提示，不报错、不生效——**静默无效**，最易被忽视）；② µs 级 kernel 的 autotune 用 do_bench 选配置，测量噪声大于配置间真实差异，会系统性误选。小 kernel 直接用静态 BLOCK 规则；45-case 级多 shape 场景宁可按 shape 分段写 if-else 静态规则，也不要 autotune。>

### 实测数据（三组独立证据）

| 组 | 现象 | 数据 |
|---|---|---|
| 直给卡组 | opt 轮挂 autotune（multibuffer 开关） | 4.1228x < 静态轮 4.2060x，拒绝 |
| 无知识组 | autotune 误选记录 | 「µs 级 kernel do_bench 噪声误选」列入勿重试清单 |
| KB 检索组 | 最终代码保留 autotune | 34/45 case 显著慢于静态组（per-shape 配置被噪声带偏），geomean 低 15% |

num_stages 不生效的识别：运行时 WARNING（易被日志淹没）；IR 层面无 multibuffer 结构生成。库内历史卡 [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) 源自 GPU/旧版经验——**在 Triton-Ascend 当前版本上以本卡为准**。

### 替代做法

- 静态规则示例（45-case 扫掠标定）：`BR = min(32, 4096//BD) if R>=768 else (8 if BD>=128 else 16)`——按 shape 分段，编译期确定；
- 若必须 autotune（大 kernel）：固定 do_bench 的 warmup/rep 到足够大、对选中 config 人工复核一遍 benchmark，且 key 覆盖全部运行时维度。

<!-- okf:related:start -->

# 相关

- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 主题相关：共享 autotune
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](../compute-pipeline/for-loop-tiling-ilp.md) — 主题相关：共享 num_stages
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 autotune
- 相关主题: [跨 run benchmark 噪声鉴别纪律（framework 侧 3-7x 波动，speedup 跨 run 对比无效）](../meta-skills/cross-run-benchmark-noise.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 同属优化知识，主题相关
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](autotune-debug.md) — 主题相关：共享 autotune
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) — 主题相关：共享 autotune, num_stages, num_warps
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](tiling-candidate-compile-cost-gate.md) — 同为 autotune 不适用、静态规则/host 公式替代的主题卡

<!-- okf:related:end -->
