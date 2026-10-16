---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: autotuning
dimension: "5.1"
title: Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune
description: tiling 候选扫描前先按编译成本预过滤——实测 BT=64（大 chunk 内 token 维）编译 >400s 且 aicore trap，调参空间被实证封死；当 BLOCK 参数可由 shape 公式确定性推导（如 BK=next_pow2(K)、BV=min(BVP,64)）时视为 autotune 不适用，host 分档优于搜索。
tags: [compile_cost, search_space_pruning, block_size, host_formula, autotune_inapplicable, chunked_recurrence]
confidence: verified
status: draft
created_at: "2026-09-14T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

## CT-1 Tiling 候选编译成本闸

【优化维度】：调优｜搜索空间
【泛化层级】：条件性（迁移条件：候选含超大分块档位或 kernel 分块参数多维耦合；失效边界：编译时间正常（秒级）的候选不受本闸约束）

<原则：autotune/候选扫描的每个 config 都要付一次编译成本；超大分块档位（如 chunk 内 token 维 BT=64）编译 >400s 且触发 aicore trap，实测收益空间为零。搜索前必须用编译成本上限预过滤候选档位；若 BLOCK 参数可由 shape 公式确定性推导（BK=next_pow2(K)、BV=min(BVP,64) host 分档），按规则视为 autotune 不适用，禁止盲目搜索。>

**约束 → 违反后果 → 满足方式**：

- **约束**：单候选编译时间有上限预期（秒级）；多维分块参数（BK/BV/BT/BVP/NVB 并存）不满足单维 scaling 类自动调优的触发前提。
- **违反后果**：编译爆炸（>400s/档）+ aicore trap 伪失败，调参预算全部烧在无效档位上。
- **满足方式**：候选生成时剔除触发编译爆炸的档位；多维分块参数改 host 公式分档；验证"调参空间封死"用编译成本实证而非逐一试跑。

**已知实例**：Gdn2Fwd（chunk 递推 + CV 融合，ascend910b1）——BT=64 编译 >400s + aicore trap，BT 可调空间实证封死；BK/BV host 按 shape 公式分档后 autotune（点 32）与 Block Size Scaling（点 33，多维分块参数不满足触发条件）均不适用，全趟扫描收敛（2026-09-14；证据：op_94_94_Gdn2Fwd_20260914_1219_8124/output/opt_iter_3/log.md + report.md「优化历程」）

**已知实例（二次确认）**：Gdn2Fwd（chunk 递推 + CV 融合，ascend910b1）——独立二次生成复现同一结论：BK/BV host 按 shape 公式分档 → autotune（点 32）不适用；BT=32 维持不变（BT=64 编译爆炸 + BT=16 劣化的既有实证未被推翻）；多 kernel 均含多维分块参数（BT+BK+BV / BK+BV）→ Block Size Scaling（点 33）触发条件 3 不满足；全趟扫描收敛（scan_complete）后 geomean 31.2788（2026-09-16；证据：op_94_Gdn2Fwd_20260916_1619_2077/output/opt_iter_2/log.md + summary.json）。

<!-- okf:related:start -->

# 相关

- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — Gdn2Fwd 回流 CT：chunked_recurrence 大 chunk 档位编译爆炸实证
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — 本案例蒸馏出的 CT 卡：BT=64 编译爆炸 + host 公式分档替代 autotune
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 共享 chunked_recurrence：大 chunk 档位编译成本约束影响分档
- 相关主题: [Compute Optimization: compile time dispatch](../compute-pipeline/exp_radix_convert_001.md) — host 公式分档与 compile time dispatch 同属确定性分派
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — BLOCK 选择约束族：本卡补编译成本维度（UB 预算之外）
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 按 shape 特化 autotune 配置与 host 公式分档同主题
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — autotune tile 扫描实践需过本卡编译成本闸
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](autotune-debug.md) — autotune 使用陷阱族：本卡补编译成本闸维度
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) — autotune 配置空间主题：本卡讲空间被编译成本封死/不适用时的替代
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](exp_perf_017.md) — 调优空间设计主题：tile/编译成本同为候选预过滤变量
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](triton-ascend-microkernel-autotune-noise.md) — 同为 autotune 不适用、静态规则/host 公式替代的主题卡

<!-- okf:related:end -->
