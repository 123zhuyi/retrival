---
schema_version: okf.v1
kind: runbook
type: implementation_paradigm
source_family: curated
title: delta-rule 族 chunk WY 实现范式（串行递推核的 chunk 化突围）
description: 逐 token delta-rule 递推核（GDN-2/KDA/GDN v1/Mamba 类线性注意力）撞上 per-token pipe 同步地板时的标准突围路径：chunk 内 WY 三角求解 + 子块参考点防 exp 上溢 + 成本模型 dispatch。含 template 臂实测参数、我方代数推导与 simulator 动机实证。提取时判「WY 不可实现/不值得」前必读。
tags: [delta_rule, gdn, kda, linear_attention, wy_representation, chunk, triangular_solve, cumsum, dispatch, recurrent_state]
confidence: inferred
status: draft
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

# delta-rule 族 chunk WY 实现范式

> 适用判定：算子是逐 token delta-rule 状态递推（S_t = (I − k(βk)ᵀ)Diag(α)S_{t−1}
> + k(v_new)ᵀ 形态），且串行骨架（逐 token 向量递推）的 simulator 采集显示
> **同步类（BAR+SET_FLAG+WAIT_FLAG）占比 ≥40%**——此时 WY 是已证有效路径，
> **禁止用「op 量对消/实现复杂」等 [DERIVED] 分析在假设阶段否决**
> （94 的教训：该否决被实测推翻，代价 2.54x 差距的三分之一）。
>
> 证据等级：[EXP] = template 臂 94_Gdn2Fwd opt5-10（2026-09-14，910b1）；
> [EXP-skeleton] = 本方骨架 simulator（2026-09-17，T=64 上下文）；
> [DERIVED] = 代数推导（无 fla 源码可锚）。

## 1. 动机（什么时候必须上 WY）

逐 token 串行核的结构性地板：每 token 循环 5 个同步点，同步类 cycles 占比
**51.1% [EXP-skeleton]**（template 臂记录 ≈56% [EXP]）——向量管线的访存/计算
无法跨 token 重叠，长 T 时该地板主导总时长。**chunk 化把 BT 个 token 的递推
改写成矩阵运算（Cube 利用 + 跨 token 并行），是打破地板的唯一已证路径**。

收益实测 [EXP]（94_Gdn2Fwd，50-case）：dispatch 组 ×1.88、K1 直读+cumsum
×1.4549、长 T case 全面反超串行核（T≥250 桶 impl 比 0.600 → WY 启用后
预计 0.85+）。

## 2. 代数结构（[DERIVED]，94 extraction §6.x 全推导）

逐 token 五步递推（以 GDN-2 为例：衰减→擦除读→delta→秩一写→读出）按
chunk（BT 个 token）重排：

```
v_new = (I + tril(A,−1))⁻¹ · u          ← BT×BT 严格下三角前代求解（WY 表示）
    A[j,l] = β_jᵀ k̄_l (j>l)；β_j = b_j⊙k_j⊙exp(ĝ_j)；u_j = w_j⊙v_j − β_jᵀS₀
S_end = Diag(exp(ĝ_BT))·S₀ + K̃ᵀ·V_new   ← 单个 matmul 合成 chunk 末状态
    K̃_j = k_j ⊙ exp(ĝ_BT − ĝ_j)
o_i = scale·[q̃_iᵀS₀ + Σ_{j≤i}(q_iᵀDiag(exp(ĝ_i−ĝ_j))k_j)·v_new_j]
```

其中 ĝ 为 chunk 内 inclusive cumsum（逐通道，[K] 向量）。

## 3. 数值红线：exp(−ĝ) 上溢与子块参考点分解（[DERIVED] + [EXP] 佐证）

- g∈[−5,−0.1]/token 时，BT=64 的 ĝ 可达 −320，**exp(−ĝ_l) 直接物化必上溢**
  （fp32 上限 ~e^88）——这是 WY 被误判「不可实现」的头号理由，**有标准解法**：
- **子块参考点分解**（mamba2 SSD 同款）：chunk 划 16-token 子块，块 q 内逆衰减
  相对块末参考点 ĝ_{q_end} 计算（ĝ_j − ĝ_{q_end} ≤ 0，指数 ≤1 安全）；跨块项
  ĝ_i − ĝ_{q_end} ≤ 0 同样安全。**所有指数因子恒 ≤1，上溢结构性消除**；
- 对角子块内 A 的逐通道 pairwise 和无法用单 matmul 表达（衰减逐通道）：
  [16,16,BK_tile] 广播乘 + K 向归约，或子块内退回逐 token 递推。

## 4. 实现参数（全部 [EXP] template 臂 opt5-10 实测，勿重新标定默认值）

| 参数 | 值 | 依据 |
|---|---|---|
| BT（chunk 大小） | **32** | BT=16 证伪：K1 Neumann 工作量∝BT²、K2 串行步数翻倍（opt8）；勿恒 64 |
| K1（chunk 内准备） | **[BT,BK] 朝向直读原布局 + tl.cumsum** | 消除 host 侧 6 个预转置 permute+contiguous（opt9，chunk 组 ×1.4549） |
| K2（跨 chunk 串行） | 状态 [BK,BV] 常驻，chunk 间递推 | 同 serial 骨架 |
| dispatch 守卫 | serial grid≥20 留在串行核；`avg_seg≥192 && n_seg·H·NVB≤20` 并集转 chunk | opt5/opt10（新转 chunk 组 ×1.88；27/40 重路由 ×1.39/×1.49） |
| BV | 64（[BK,64] fp32 ≤32KB 编译边界） | template L1.8 + 本方 P2.5 编译门 [EXP-skeleton] 双证 |

## 5. 勿重试清单

1. **「op 量对消」否决 WY**：[DERIVED] 分析（三角求解增加的算力 ≈ chunk 化省
   下的串行步）被实测推翻——同步地板的消除收益不出现在 op 计数里（94 首轮
   教训，[DERIVED] 权重 < 同结构实测）；
2. **exp(−ĝ) 无参考点直接物化**：fp32 上溢（§3）；
3. **BT=16 或恒 BT=64**：前者 ∝BT² 工作量爆炸 [EXP opt8]；后者 ĝ 范围加深
   上溢压力且短段空转；
4. **host 侧预转置**：K1 改 [BT,BK] 直读可全消（[EXP opt9]——host 6 个
   permute+contiguous 是 SpyKernel 墙钟分解确认的热点）；
5. **全量转 chunk（无 dispatch）**：短 T/大 grid case 串行核更优（serial
   grid≥20 守卫 [EXP opt10]）。

## 6. 遗留开放项（诚实记账）

- 本方谱系**尚未实现过 WY 骨架**（94 v1.1 复跑 agent 以成本收益决策未试，
  当时 target 已六倍达成）——本卡的实现参数全部来自 template 臂 [EXP]，
  首次在本方落地时按新算子对待（P2.5 编译门+正确性门先行）；
- tl.cumsum 在 triton-ascend 的 lowering 分级未入 tl_op_lowering 表——
  首次使用前查 backend_facts，缺则按 L2/L3 检索补齐（P2.5 ④ IR 门可验证）；
- 对角子块的逐通道 pairwise 和的最优 Triton 形态（广播+归约 vs 子块内逐
  token）未两态。

## 维护记录

- 2026-09-17 初版：素材 = template 臂 94_Gdn2Fwd opt5-10 实测参数（report.md
  版本谱系表）+ 本方 extraction §6.x 代数推导 + P2.5 simulator 动机实证
  （p25_pilot/pilot_report.md）。立项动因：用户指出「提取时误判功能点不可
  实现（实际可实现），实现范式应提取为公共卡」——WY 是首个按此立法的范式。
