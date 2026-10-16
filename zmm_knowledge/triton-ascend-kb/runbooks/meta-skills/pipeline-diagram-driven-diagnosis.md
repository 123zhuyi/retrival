---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.1"
title: 流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段
description: 优化 CV 融合/递推类算子前，先取指令流水图（Triton）或泳道图+计算图（PyPTO），把气泡按"块内 CV 等待边 / 跨 chunk 时序边 / 搬运空闲 / 异常碎块"分类，每类气泡映射到固定手段集；GDN 与 PyPTO 求逆模块两个案例互证该方法论。
tags: [pipeline_diagram, swimlane, dfx, bubble_classification, profiling_first, methodology, gdn, pypto]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-08-27T00:00:00Z"
---

## 流水图驱动优化方法论（来源：GDN AscendC 流水图 + PyPTO 泳道图 DFX 两案例互证）

【优化维度】：元技能｜瓶颈定位
【泛化层级】：通用（迁移条件：任何多模块流水化 kernel；失效边界：单模块纯 elementwise kernel 无需流水分析，直接看访存）

<原则：流水类算子的优化手段与瓶颈位置强耦合——同一招放错位置收益为零甚至为负。动手前先可视化：Triton 侧取指令流水图（Cube/Vector/MTE 各 pipeline 时序），PyPTO 侧用泳道图（核间/核内任务时序）+ 计算图（拷入拷出节点）。把每个气泡归类，再按类取招。>

**气泡分类 → 手段映射表**：

| 气泡类型 | 图特征 | 首选手段 |
|---|---|---|
| 块内 CV 等待边（C→V 或 V→C 串行） | Cube 与 Vector 条带首尾相接、无重叠 | 多重下发 + 双缓冲（multibuffer/tile_mix 配置） |
| 跨 chunk 时序边（后块 C 等前块 V） | 每轮迭代起点被上一轮尾部长任务卡住 | Pingpong 下发顺序重排 + Head/任务交错（见 compute-pipeline Pingpong 卡） |
| 搬运空闲（MTE3 搬出时计算全停） | 计算条带在搬出段整体空缺 | 输出掩盖：下一 chunk 任务提前开始；预加载独立输入 |
| 带宽竞争（关键路径被非关键写出拖慢） | 关键 C/V 边异常拉长 | Triton：推迟非关键 store；AscendC：晚读出仲裁 |
| 异常碎块/超长块（PyPTO 特有） | 大量极短碎块 + 个别异常长块 | 查计算图：合切图是否符合前端表达；拷入拷出是否冗余 |

**案例 A（AscendC 流水图）**：Fwd_h 流水图红框定位"第三个 C1 等第一个 V2"→ 确认是跨 chunk 边 → 下发顺序 C1C1C2C1 改 C1C1C2C2 → 910B1 相对朴素方案 +41%。若误用块内手段（加双缓冲），该气泡不消失。

**案例 B（PyPTO 泳道图+计算图）**：求逆模块 8×[16,16] 并行求逆，泳道图显示碎块多、合并尾轴为 2×[16,64] 后收益不及预期（8×22μs→2×62μs）→ 计算图发现绿色拷入/红色拷出节点过多：逐行更新的矩阵在 UB 频繁搬进搬出 → 改为常驻 UB、仅首尾各一次拷贝 → 进一步 concat 为 1×[16,128] → 8×28μs→49μs。**教训：穿刺收益不及预期时，不要怀疑方向，先查计算图的搬运节点。**

**操作流程**：

1. 跑 baseline 取流水图/泳道图；
2. 标出所有 >5% 占比的气泡与异常块，逐条归类到上表；
3. 按类选招、一次改一类，每步消融验证（GDN 实践全部为消融驱动，避免了手段间的归因混淆）；
4. 残余瓶颈若落在"Triton 不可达区"（带宽仲裁、RegBase 级 UB 消除），触发 AscendC 升级判据（见 ascend-950 平台卡）。

**注意事项**：

- 没有流水图时禁止凭代码直觉上 Pingpong/双缓冲——GDN 消融显示同一手段（PingpongA）在 950PR 变长场景出现回退（226.66→271.25ms），盲目套用有害。
- 泳道图还能直接区分 vector bound / cube bound，决定后续是加 Vector 并行还是优化 Cube 链路。

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — Fwd_h 为流水图气泡分类方法论的目标算子
- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 gdn, pypto
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — per-token BAR≈56% 结构性地板即流水/同步分析归因
- 实践案例: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 同属优化知识，主题相关
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 gdn
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 gdn
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../compute-pipeline/load-balance-grid-sizing.md) — 主题相关：共享 gdn
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 gdn
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 主题相关：共享 gdn
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
