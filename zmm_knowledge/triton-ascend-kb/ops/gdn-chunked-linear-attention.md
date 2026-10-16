---
schema_version: okf.v1
kind: operator
type: case_study
source_family: ops
title: GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）
description: Gated Delta Network 在昇腾上的完整优化轨迹：Triton 移植层手段（分核/Tiling/layout/mask）带来近百倍开箱提升，PyPTO MPMD 路线验证调度配置价值，AscendC 重写针对 Fwd_h/Fwd_o 两个热点以 Pingpong+流水+950 特性再超 Triton 基线 1.1~4 倍；是 chunk 递推+CV 融合类算子的方法论样板。
tags: [gdn, linear_attention, gated_delta_net, chunked_recurrence, cv_fusion, case_study, fla, pypto, ascendc, triton_ascend]
confidence: draft
status: draft
resource: https://jx.huawei.com/community/comgroup/postsDetails?postId=1b611498942d4a1c85f99023cabaaf7a
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

sources:
  - url: https://jx.huawei.com/community/comgroup/postsDetails?postId=1b611498942d4a1c85f99023cabaaf7a
    role: primary

## GDN 三阶段优化案例（来源：2025-12 / 2026-03 / 2026-08 三篇内部实践）

【案例类型】：chunkwise 递推 + CV 融合算子族（GDN/KDA/Mamba2 同构）
【价值定位】：本 KB 多张 technique 卡的共同出处与收益数据源；遇到线性注意力/chunk 递推类任务时先读本卡建立全局图，再按需取卡。

### 算法与 kernel 映射（chunkwise 形式）

GDN 状态更新：Sₜ = αₜSₜ₋₁(I−βₜkₜkₜᵀ) + βₜvₜkₜᵀ（Mamba2 门控遗忘 + DeltaNet 定向纠错的结合）。FLA 实现对应 6 个 kernel：

| 数学项 | kernel | 角色 |
|---|---|---|
| Γ（log  decay 累积） | chunk_local_cumsum | 前置 |
| β·(Γ⊙KKᵀ) | chunk_scale_dot_kkt_fwd | 前置 |
| (I+tril(...))⁻¹ | solve_tril | 块内求逆 |
| U, W | recompute_w_u_fwd | 前置 |
| **S[t+1]** | **chunk_gated_delta_fwd_h（Fwd_h）** | **热点：块间时序依赖** |
| **O[t]** | **chunk_fwd_kernel_o（Fwd_o）** | **热点：仅块内 CV 依赖** |

Fwd_h 与 Fwd_o 的关键差异：Fwd_h 的 S_new 是下一块输入（块间串行 + 并行度受限），Fwd_o 无此依赖——优化手段因此分叉。

### 三阶段演进与手段→卡的映射

**阶段一（Triton 移植层，"迁移四板斧"）**——相对 FLA 开箱近百倍：

| 手段 | 对应卡 | 代表收益 |
|---|---|---|
| grid 对齐物理核数 | staging/load-balance-grid-sizing.md | fwd_o/kkt +99% |
| BV/BK 64→128 打满 UB，微超则外提 mask/g_exp | staging/ub-fill-and-compute-hoist.md | dqkwg 18→10ms |
| layout (B,T,H,D)→(B,H,T,D) | staging/layout-bthd-to-bhtd.md | fwd 平均 +38%~79% |
| int64 索引转 fp32 比较、冗余下三角化删除 | staging/int64-mask-index-scalar-fallback.md 等 | solver_tril 省 2ms |

**阶段二（PyPTO MPMD 路线）**：stitch 三参数（inner/outcast/num_initial）10/50/30→128/128/128 提并行度；Tile [64,128]→[128,128]；动态 Loop 改静态 + unroll_list；DFX 泳道图深调求逆模块（8×28μs→49μs）。推理 S=4K：23.21ms→5.70ms→**0.81ms**。教训：调度配置与合切图是 MPMD 框架的一等优化变量（PyPTO 专属，不进 Triton 卡）。

**阶段三（AscendC 重写，Triton 已达平台上限后）**：保持多算子拼接模式（未走整网大融合，权衡开发复杂度），针对 Fwd_h/Fwd_o：

| 手段 | 对应卡 | 代表收益（AscendC 语境） |
|---|---|---|
| 依赖二分 + PingpongB（C1C1C2C2）+ Head 交错 | staging/chunk-recurrence-pingpong-dispatch.md | 910B1 +62% |
| 预加载 / 晚读出 / 输入复用 | staging/cv-preload-and-bandwidth-arbitration.md | +26%~58% |
| 950 CV 通路 + RegBase | staging/ascend-950-cv-datapath-regbase.md | 950DT 再 +37.5% |

最终：Fwd_h 达 Triton 基线 3~4×（910B1）、1.5×（950DT）；Fwd_o 2.7~4×（910B1）；整网单算子平均 +70%、端到端 +20%+，代码合入 flash-linear-attention-npu 并应用于 vllm-ascend。

### 复用决策要点

1. **先用阶段一四板斧**（全部 Triton 可达），通常已覆盖 80%+ 收益；
2. 残余瓶颈用流水图归类（staging/pipeline-diagram-driven-diagnosis.md）再选阶段三手段；
3. **使用建议**（定长场景）：子序列 >8K 时 GDN 优于 FA——8K 持平、32K 达 5.5~6.6×FA；<8K 场景 FA 仍优；
4. chunkSize 一律先试 128（Fwd_h 平均 +46.75%、Fwd_o +18.63%），代价是片上缓存预算；
5. 精度验证采用双标杆法（GPU Triton + CPU 单精度为参照，L1 级通过标准）。

**消费反馈（已知实例）**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1，2026-09-14）设计期 strong 命中本卡——其"chunk 递推 + CV 融合算子族方法论"直接指导了 serial/chunked WY 双骨架架构选型（host 成本模型 dispatch，50 case 几何平均 22.25，大 T case 2.8-4.1×）；见 [Gdn2Fwd 双骨架 host dispatch 实测案例](gdn2-fwd-chunked-wy-dispatch.md)。

<!-- okf:related:start -->

# 相关

- 实践案例: [solve_tril 算子优化（三角矩阵求逆）](../reference/op-class-solve_tril.md) — 主题相关：共享 fla
- 实践案例: [SSM / 状态空间模型算子优化](../reference/op-class-ssm.md) — 主题相关：共享 linear_attention
- 实践案例: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](../runbooks/autotuning/tiling-candidate-compile-cost-gate.md) — Gdn2Fwd 回流 CT：chunked_recurrence 大 chunk 档位编译爆炸实证
- 实践案例: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../runbooks/compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 chunked_recurrence, gdn
- 实践案例: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../runbooks/compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 gdn
- 实践案例: [Compute Optimization: chunked scan strategy](../runbooks/compute-pipeline/exp_chunk_cumsum_001.md) — 同属优化知识，主题相关
- 实践案例: [Compute Optimization: vectorized chunked processing](../runbooks/compute-pipeline/exp_triton_rope_001.md) — 同属优化知识，主题相关
- 实践案例: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../runbooks/compute-pipeline/load-balance-grid-sizing.md) — 主题相关：共享 gdn
- 实践案例: [Memory Access Optimization: chunked recurrence strategy](../runbooks/memory-sram/exp_chunk_gated_delta_001.md) — 同属优化知识，主题相关
- 实践案例: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../runbooks/memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 chunked_recurrence, gdn, linear_attention
- 实践案例: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../runbooks/memory-sram/ub-fill-and-compute-hoist.md) — 主题相关：共享 gdn
- 实践案例: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../runbooks/meta-skills/pipeline-diagram-driven-diagnosis.md) — 主题相关：共享 gdn, pypto
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 cv_fusion, gdn, linear_attention
- 相关主题: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](gdn2-fwd-chunked-wy-dispatch.md) — Gdn2Fwd 设计期 strong 命中本卡方法论：chunk 递推+CV 融合族架构选型
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 linear_attention

<!-- okf:related:end -->
