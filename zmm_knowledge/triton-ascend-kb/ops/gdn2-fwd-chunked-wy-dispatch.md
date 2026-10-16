---
schema_version: okf.v1
kind: operator
type: case_study
source_family: ops
title: Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY
description: Gated DeltaNet 2 chunkwise 前向（varlen + 可选 initial_state + kernel 内 L2 归一化）在 910b1 上的实测生成案例：serial persistent 单骨架有 per-token pipe 同步（BAR+SET/WAIT≈56%）结构性地板，host 侧成本模型分派 serial（小 case）/ chunked WY 双 kernel（大 T case）后 50 case 几何平均加速比 22.25（vs PyTorch eager 现场实测），大 T case 提速 2.8-4.1×；Phase 4 全量扫描（点 11/28/31/32/33）证伪后性能收敛。2026-09-16 二次独立验证 31.28（serial double-buffer ×1.404 + dispatch ×1.051）。适用于 chunk 递推 + CV 融合类算子的骨架选型与优化终局判断。
tags: [gdn, gated_delta_net, linear_attention, chunked_recurrence, wy_representation, varlen, dual_skeleton, host_dispatch, cost_model, case_study, ascend910b1, triton_ascend]
confidence: verified
status: draft
resource: file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_94_94_Gdn2Fwd_20260914_1219_8124/report.md
created_at: "2026-09-14T00:00:00Z"
updated_at: "2026-09-16T00:00:00Z"
---

sources:
  - url: file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_94_94_Gdn2Fwd_20260914_1219_8124/report.md
    role: primary
  - url: file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_94_Gdn2Fwd_20260916_1619_2077/report.md
    role: primary

## 实测记录（verified run）

- **算子**: 94_Gdn2Fwd（Gated DeltaNet 2 chunkwise 前向，varlen + 可选 initial_state + kernel 内 L2 归一化；category: linear-recurrent）
- **arch**: ascend910b1 | **生成任务**: op_94_94_Gdn2Fwd_20260914_1219_8124 | **日期**: 2026-09-14
- **精度**: 50/50 shapes 全过（证据：`output/iter_0/verify/verify_result.json`）
- **几何平均加速比**: 22.2468（口径：vs PyTorch eager 现场实测，未注入 perf.json；证据：`summary.json#perf_data.speedup_vs_torch`）
- **目标达成**: target_speedup=5.0，target_reached=true
- **基线口径警示**：per_shape 的 speedup 为「该 shape vs 全局 geomean」口径，仅顶层几何平均有对比意义

代表性 case（全 50 case 指回 `summary.json#perf_data.per_shape_results`）：

| case | shape (q) | fw (ms) | impl (ms) | speedup |
|---|---|---|---|---|
| 15 | [1, 593, 6, 41] | 30.6438 | 0.3114 | 98.4066 |
| 30 | [1, 1578, 8, 24] | 60.3231 | 0.6540 | 92.2372 |
| 38 | [1, 3388, 12, 41] | 158.8728 | 1.8203 | 87.2784 |
| 43 | [1, 2301, 8, 53] | 110.4836 | 0.9971 | 110.8049 |
| 48 | [1, 759, 12, 72] | 47.3471 | 0.6470 | 73.1794 |
| 1 | [1, 26, 16, 37] | 1.3603 | 0.0508 | 26.7776 |

（上 5 行为 chunked 路径大 T case，相对 serial 同 case 提速 2.8-4.1×；末行为 serial 路径小 case 代表。）

## 架构与关键决策

双骨架 + host 成本模型 dispatch（无 D2H 数据依赖，len/shape 推导）：

| 骨架 | kernel | 适用 case | 依据 |
|---|---|---|---|
| A serial persistent | `_gdn2_serial_kernel`，grid=(n_seg·H·NVB,) | 45 个小 case | 状态 [BK,BV] fp32 驻留 UB 逐 token 递推；per-token pipe 同步（BAR+SET/WAIT≈56%）是结构性性能地板 |
| B chunked WY 双 kernel | K1 `_gdn2_wy_prep_kernel`（grid=(n_seg·C_max·H,)，并行：dot-cumsum→Gram(mid-row)→Neumann 5 因子→fp32 工作区）+ K2 `_gdn2_chunk_fwd_kernel`（grid=(n_seg·H·NVB,)，chunk 串行，每 chunk 4 大 dot） | 5 个大 T case | WY 表示消解块间时序依赖，块内并行度释放 |
| host dispatch | `use_chunked = (n_seg*H <= 15) and (T >= 500)` | — | 成本模型：chunked 2.85us·C·H + 5.5us·T/32·waves + 150us host vs serial 1.7us·T |

关键决策：
1. **大 T 必须并列评估 chunked/WY 骨架**：serial 骨架 per-token pipe 同步占比 56% 是结构性地板，调参无法消除（IR 证伪）。三方对照：纯 serial 19.91 geomean / 无 dispatch 纯 chunked 口径失真 / 双骨架 dispatch **22.25**，iter_0 一次生成即超 serial 天花板 +11.7%（公共 48 case impl 延迟几何比 1.1046）。
2. **3-exp 分解数值必需**：e_hi = eG/exp(g_ref) 的三指数分解在强衰减下 0/0=NaN，不可合并为单 exp（IR 轮证伪合并建议）。
3. **状态 store 纪律**：masked 2D state store 静默丢单元格（头号精度陷阱），必须 flat 1D 布局写回或逐行 store；store mask 必须 int32 比较（fp32 导出 mask 并发下非确定性 store 污染）——见 `runbooks/portability/triton-ascend-mask-codegen-traps.md`。
4. **UB 预算边界**（910b1, UB=192KB）：[128,128] fp32 状态 + 输入向量编译报 ub overflow；[128,64] fp32（32KB）通过且为性能最优点（BV cap=64 平衡点，cap=32 劣化 24%）。
5. **数值策略**：状态、门控衰减、归约全程 fp32，输入读入即 cast fp32，仅最终 store cast 回目标 dtype。

## 优化历程（Phase 4 摘要，4 轮全判定）

| 轮 | 命中点 | 结果 | 归因 |
|---|---|---|---|
| opt_iter_0 | 11 Load 重排序 | 无提升（22.0006 vs 22.2468） | IR 显示 K2 dot 链关键路径在 Cube 侧，load 重排未改变关键路径 |
| opt_iter_1 | 28 CV 融合 | 平台不可用 | `triton.language.extra.ascend` 仅 libdevice，无低级 CV API（al./bl.），import 即失败（triton 3.2.0 + 910b1） |
| opt_iter_2_ir_1 | 31 IR 分析 | ir_has_more_suggestions=false | K1 12 matmul 全 ieee fp32（数值红线）；3-exp 数值必需；K2 4 matmul+双 L2N 数学固有 |
| opt_iter_3 | 32/33 终止步骤 | scan_complete=true | BK/BV host 按 shape 公式分档视为不适用 autotune；BT=64 编译 >400s + aicore trap 实证封死；多维分块参数不满足 Block Size Scaling 触发条件 |

## 复用决策要点

1. chunk 递推类算子（GDN/KDA/Mamba2 同构）从 serial 起步可行，但 **n_seg·H≲15 且 T≳500 时必须并列评估 chunked WY 骨架**，host 成本模型分派（无 D2H 依赖）是零运行时开销的正确形态；
2. serial 骨架的 per-token BAR 占比是结构性指标：BAR+SET/WAIT ≈56% 时调参与 load 重排均无收益，直接转骨架评估；
3. 低级 CV API（al./bl.）在 910b1 + triton 3.2.0 不可用，CV 融合类优化点在该平台直接判死，勿再尝试；
4. BT（chunk 内 token 维）不能盲目调大：BT=64 编译 >400s 且 aicore trap，调优空间须过编译成本闸；
5. 状态矩阵 fp32 驻留 UB 的预算边界实测：910b1 192KB 下 [128,64] fp32 是 [BK,BV] 性能最优点，[128,128] 溢出；
6. 精度验证环境红线：`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 必设，否则双后端注册冲突产生大面积伪失败。


## 实测记录（第二次独立验证，2026-09-16）

- **算子**: 94_Gdn2Fwd | **arch**: ascend910b1 | **生成任务**: op_94_Gdn2Fwd_20260916_1619_2077
- **精度**: 50/50 shapes 全过（证据：`output/iter_1/verify/verify_result.json`）
- **几何平均加速比**: **31.2788**（口径：vs PyTorch eager 现场实测，未注入 perf.json；证据：`summary.json#perf_data.speedup_vs_torch`）
- **目标达成**: target_speedup=5.0，target_reached=true
- **代码谱系**: 与首次 run 独立——从用户指定 skeleton v1.1 + 本卡守卫（`use_chunked = (not varlen) and (N*H <= 15) and (T >= 500)`，verbatim 复用）起步，非首次 run 代码续作

**Phase 4 优化历程（4 轮全判定）**：

| 轮 | 命中点 | 结果 | 归因 |
|---|---|---|---|
| opt_iter_0 | 11 Load 重排/double-buffer | 21.2027 → **29.7598**（×1.404 采纳，verify 50/50） | serial persistent 骨架访存流水化——**与首次 run 点 11 负结果并存**：同优化点在 serial 骨架（访存侧气泡）有效、在 chunked K2（Cube 侧关键路径）无效，失效边界在骨架不在优化点 |
| opt_iter_1 | 12 Grid 多路径特化 | 29.7598 → **31.2788**（×1.051 采纳） | chunked WY dispatch 命中 5 个 dense 长 T case：15 ×1.905 / 30 ×1.543 / 38 ×1.587 / 43 ×1.871 / 48 ×2.100，组几何 ×1.789（与首次 run ×1.45-1.88 区间一致） |
| opt_iter_2_ir_1 | 31 IR 分析 | ir_has_more_suggestions=false | K2 循环内 199 sync ops（set_flag 64 + wait_flag 67 + pipe_barrier 59 + sync_block_set/wait 19+19）对应 8 dot/chunk + 向量 op 的 CV 交替，结构性地板；K1 12 dot + K2 8 dot 全 ieee fp32 数值红线；循环内 1 个 arith.remsi 为编译器把 `a-(a//b)*b` canonicalize 回 rem（标量 task 解码，可忽略） |
| opt_iter_2 | 32/33 终止步骤 | 均不适用 → scan_complete | 与首次 run 结论一致：BK/BV host 公式分档 → autotune 不适用；BT=32 空间实证封死；多 kernel 均含多维分块参数（BT+BK+BV / BK+BV）→ 点 33 触发条件 3 不满足 |

**Phase 3 精度修复闭环**：iter_0 final_state 写回采用 `tl.reshape` 版 flat 1D store → 后端误编译，fs 丢格 53344/81696 且非确定复现（verify 0/50，o 全部正确）；iter_1 回退 2D masked store（int32 比较 mask）→ 50/50 全过。A/B 实验证明 2D masked store 与逐行 store 均确定性正确（diff 9e-8），唯 reshape flat 1D 触发误编译——对 `runbooks/portability/triton-ascend-mask-codegen-traps.md` §4「正解：flat 1D」构成本后端（triton 3.2.0 + 910b1）边界细化。

**开放项**：varlen 长 T case（18/28）仍走 serial 未受益（varlen chunked 需段内 chunk 对齐）。

证据：`summary.json`、`report.md`、`output/opt_iter_0/{log.md,optimized_perf_result.json}`、`output/opt_iter_1/{log.md,optimized_perf_result.json}`、`output/opt_iter_2_ir_1/log.md`、`output/opt_iter_2/log.md`、`output/iter_0/log.md`（工作目录 `op_94_Gdn2Fwd_20260916_1619_2077`）

<!-- okf:related:start -->

# 相关

- 实践案例: [SSM / 状态空间模型算子优化](../reference/op-class-ssm.md) — GDN/线性注意力属 SSM 算子类
- 实践案例: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](../runbooks/autotuning/tiling-candidate-compile-cost-gate.md) — 本案例蒸馏出的 CT 卡：BT=64 编译爆炸 + host 公式分档替代 autotune
- 实践案例: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../runbooks/compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — Gdn2Fwd 选型 strong 命中本卡：依赖二分用于双骨架论证（消费反馈）
- 实践案例: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../runbooks/compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — Gdn2Fwd 点 11 负结果（Cube 侧关键路径）为失效边界已知实例
- 实践案例: [Scheduling Optimization: persistent kernel grid stride loop](../runbooks/compute-pipeline/exp_linear_persistent_001.md) — serial persistent 骨架对应 persistent kernel grid-stride 模式
- 实践案例: [Compute Optimization: compile time dispatch](../runbooks/compute-pipeline/exp_radix_convert_001.md) — host 成本模型分档 dispatch 与 compile time dispatch 同族
- 实践案例: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../runbooks/compute-pipeline/load-balance-grid-sizing.md) — 共享 gdn：核数对齐分核适用双骨架
- 实践案例: [persistent kernel 模式（grid-stride 摊还启动开销）](../runbooks/compute-pipeline/persistent-kernel-pattern.md) — serial persistent 骨架即 persistent kernel 模式（含 BAR 同步地板的反例边界）
- 实践案例: [Memory Access Optimization: chunked recurrence strategy](../runbooks/memory-sram/exp_chunk_gated_delta_001.md) — chunked recurrence 策略与 chunked WY 骨架同构
- 实践案例: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../runbooks/memory-sram/layout-bthd-to-bhtd.md) — chunkwise 递推 layout 预转置适用 Gdn2Fwd
- 实践案例: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../runbooks/memory-sram/ub-fill-and-compute-hoist.md) — 共享 gdn：UB 预算驱动分块（[128,64] 最优即 UB 感知选择）
- 实践案例: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../runbooks/meta-skills/multi-strategy-dispatch.md) — Gdn2Fwd 双骨架 host 成本模型 dispatch 是本卡原则的实测案例
- 实践案例: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../runbooks/meta-skills/pipeline-diagram-driven-diagnosis.md) — per-token BAR≈56% 结构性地板即流水/同步分析归因
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 同 GDN chunk 递推前向算子族：Gdn2Fwd 为其 varlen GDN2 变体实测案例
- 相关主题: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](gdn-chunked-linear-attention.md) — Gdn2Fwd 设计期 strong 命中本卡方法论：chunk 递推+CV 融合族架构选型
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 同 GDN 算子族：RGDR 为逐步形式，Gdn2Fwd 为 chunkwise WY 形式

<!-- okf:related:end -->
