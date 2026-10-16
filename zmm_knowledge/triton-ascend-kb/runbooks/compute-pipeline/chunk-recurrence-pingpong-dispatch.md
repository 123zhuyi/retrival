---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.2"
title: chunk 递推算子的依赖二分法与 Pingpong 下发调度
description: chunkwise 递推类算子（GDN Fwd_h 等）同时存在块内 CV 交替依赖（C1→V1→C2→V2）与块间状态时序依赖（S_new 为下一块输入）；通过依赖二分定位气泡来源，用连续双 C1/双 C2 的 Pingpong 下发顺序 + 交错相邻 Head 规避时序依赖，相对朴素 Pingpong 再提升 24%~41%。
tags: [pingpong, chunked_recurrence, cv_dependency, pipeline_bubble, gdn, dispatch_order, head_interleave]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

## chunk 递推算子的依赖二分法与 Pingpong 下发调度（来源：AscendC GDN Fwd_h 消融实验）

【优化维度】：计算｜流水掩盖 + 任务调度
【泛化层级】：条件性（迁移条件：chunkwise 递推结构——块内 Cube/Vector 模块交替依赖 且 块间串行传递状态 S；失效边界：无块间依赖的纯 chunk 并行算子如 Fwd_o，直接做多重下发即可，无需本卡的顺序重排）

<原则：先对算子做依赖二分——①块内依赖：C1→V1→C2→V2 的 CV 交替链，气泡靠多重下发 + 双缓冲消除；②块间依赖：S_new 是下一块输入，同序列 chunk 强制串行，且后块的 C1 必须等前块 V2 结束。朴素交错下发（C1→C1→C2→C1…）下第三级 C1 仍会被块间依赖卡住；改为连续下发 2 个 C1 再连续下发 2 个 C2（C1C1C2C2），同时让同一核交错处理同序列相邻两个 Head，用 Head 间独立性掩盖 chunk 间串行。>

**下发顺序对比（Fwd_h 实测）**：

```
PingpongA（朴素）: C1 → C1 → C2 → C1 → …   第三个 C1 等待第一个 V2（气泡）
PingpongB（优化）: C1 → C1 → C2 → C2 → …   块间等待被第二个 chunk 的计算填满
Vector 侧同构：    V1 → V1 → V2 → V2 → …
```

**消融数据（AscendC 实现，chunkSize=128，PingpongA→B 的增量）**：

| 平台 | PingpongA vs 基线 | PingpongB vs 基线 | A→B 增量 |
|---|---|---|---|
| 910B1 | +21.2% | +62.1% | ~+41% |
| 950PR | +7.1% | +30.8% | ~+24% |
| 950DT | +8.5% | +42.1% | ~+34% |

结论：块间时序依赖算子的下发顺序是主要变量，需结合指令流水图逐气泡定位，不能套用通用 Pingpong。

**Triton-Ascend 落地与可达性**：

- ✅ **可达**：循环体重构使相邻两次迭代尽量独立（交错 Head 维进 grid/循环），让编译器双缓冲有重叠空间；配合 `multibuffer`、`tile_mix_vector_loop`/`tile_mix_cube_loop` 编译选项 + autotune 搜索最优组合。
- ✅ **可达**：核内按 (chunk_i, head_j) 与 (chunk_i, head_j+1) 配对处理，用 Head 间独立替代 chunk 间并行——这是纯算法层重排，无 DSL 壁垒。
- ⚠️ **部分可达**：显式的"C1C1C2C2"任务图下发顺序在 Triton 中不能直接表达，只能通过循环结构间接引导编译器调度；**晚读出**（人为延迟 UB2GM 通路、带宽优先让给 UB2L1 保关键路径）在 Triton 下不可表达。
- ❌ **不可达（AscendC 兜底）**：950 的 RegBase 寄存器编程（中间结果不落 UB）。判别出口：消融分析确认剩余瓶颈为 UB 中间变量读写且访存已无法再减时，建议转 AscendC——GDN 实践中该路径在 950DT 上相对 Triton 仍有 1.5× 空间。

**诊断方法**：先取指令流水图（或 PyPTO 泳道图），按"红框气泡"定位等待边：气泡两端若是块内 CV 边 → 双缓冲/预加载；若是跨 chunk 边 → 本卡的下发顺序重排 + Head 交错。

**注意事项**：

- 多重下发的代价是额外 GM/UB 空间存中间变量，需核算 UB 预算；chunkSize=128 时双缓冲占用翻倍。
- B*H 较小（核用不满）时 Head 交错收益下降，优先做大 chunkSize + 负载均衡分核。
- 无块间依赖的算子（如 Fwd_o）用简单多重下发即可获得 ~56% 提升，叠加 chunk 间流水（下一块提前开始）可达 +79%~94%——不要用本卡的重排过度设计。

**已知实例（消费反馈）**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1，2026-09-14）设计期 strong 命中本卡——依赖二分法用于 serial/chunked 双骨架的选型论证（块间状态时序依赖 → chunked WY 消解；块内 CV 依赖 → 保留在 K2 chunk 内），大 T case 提速 2.8-4.1×；见 [Gdn2Fwd 双骨架 host dispatch 实测案例](../../ops/gdn2-fwd-chunked-wy-dispatch.md)。

**已知实例（消费反馈，二次）**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1，2026-09-16）设计/优化期再次 strong 命中本卡——依赖二分法复用于双骨架选型复核；IR 轮实测 K2 chunk 循环内 199 sync ops（set_flag 64 + wait_flag 67 + pipe_barrier 59 + sync_block_set/wait 19+19）对应 8 dot + 向量 op 的 CV 交替，量化佐证「块内 CV 依赖保留在 chunk 内」的划分与同步结构性地板（证据：op_94_Gdn2Fwd_20260916_1619_2077/output/opt_iter_2_ir_1/log.md）。


<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 gdn
- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 chunked_recurrence, gdn
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — Gdn2Fwd 选型 strong 命中本卡：依赖二分用于双骨架论证（消费反馈）
- 相关主题: [Moe Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](../autotuning/tiling-candidate-compile-cost-gate.md) — 共享 chunked_recurrence：大 chunk 档位编译成本约束影响分档
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](../memory-sram/exp_chunk_gated_delta_001.md) — 同属优化知识，主题相关
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 chunked_recurrence, gdn
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 主题相关：共享 gdn
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../meta-skills/pipeline-diagram-driven-diagnosis.md) — 主题相关：共享 gdn
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](../meta-skills/precompute-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 cv_dependency, gdn
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 同属优化知识，主题相关
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](load-balance-grid-sizing.md) — 主题相关：共享 gdn

<!-- okf:related:end -->
