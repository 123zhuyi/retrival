---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡 v2
description: v2（2026-09-15）：三组同机生成实验（A1 KB 2.3535x / A2 仅 A 卡 1.6873x / B3 v1卡+KB 1.3251x）回流修订。核心变化：grid 默认一块一 program（v1 的 persistent-core 被证伪）；fp32 输入原生 dot+ieee（bf16x3 误决策否决）；BV 改实测标定规则 BK·BV≤2048；语义明确面向评测 golden（g/gk 同 log2 域 exp2）；新增 Phase 4 已证杠杆清单与勿重试清单。
tags: [gdn, kda, gated_delta_rule, linear_attention, recurrent_state, cv_fusion, matmul, fp16, bf16, fp32_accum, varlen, chunk_gated_delta_rule_fwd_h, v2]
confidence: measured
status: draft
resource: github.com/jiangdm2024/flash-linear-attention-npu_dev @ 4c78653f47beda77681aa725c7fa49c7ce8d86a9（本地快照 zmm_source_retrieval/src/chunk_gated_delta_rule_fwd_h/）
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-15T00:00:00Z"
---

sources:
  - url: github.com/jiangdm2024/flash-linear-attention-npu_dev @ 4c78653f47beda77681aa725c7fa49c7ce8d86a9
    role: primary

# ChunkGatedDeltaRuleFwdH v2（三组生成实验回流版）

> v1 与实验档案：`output/chunk_gated_delta_rule_fwd_h/`（v1 卡 + 骨架冻结存档）；
> 三组实验完整分析：`generation_retrospective_20260915.md`；
> 族范式：`backend_facts/operator_class_patterns.md` §4（状态常驻递推核）。
> **v2 改动原则**：语义层保留 v1（已被实验验证）；硬件/调优层全部改写实测结论
> （confidence: measured），凡解析推断一律标「起点假设，须 P4 实测」。

逐 chunk 递推线性注意力的隐状态：`v_new = u − w@h_prev`；
`h_next = h_prev·gate_decay + kg^T@(v_new·decay)`，输出每 chunk 起始状态 h、修正 v_new、
可选 final_state。算子类别：**状态常驻递推核**（每 task 大块常驻 tile + task 内长串行
循环；形态像 CV 融合，**不适用** CV 融合的 persistent-core 默认，见 §4）。

## 1. 功能与公式

- 计算公式（每 batch、v-head hv、chunk i）：

$$\text{v\_new}[t] = u[t] - w[t]\, h_i, \qquad \text{decay}[t] = 2^{\,g[\text{last}]-g[t]}$$
$$h_{i+1} = h_i \cdot 2^{\,g[\text{last}]} \cdot 2^{\,g_k[\text{last},:]} + k_g^{\top}\big(\text{v\_new}\odot \text{decay}[:,None]\big)$$

- ⚠️ **语义面向评测 golden，与 AscendC 源码有一处分叉**：
  | 项 | 85_/kernelbench Model（评分 ground truth） | AscendC 源码（GDN 分支） |
  |---|---|---|
  | g 的域 | **log2 域**（`torch.exp2`，Model:74-80 ✅） | 自然对数域（`exp`） |
  | gk 的域 | log2 域（exp2，Model:82-83 ✅） | log2 域（exp2=×LN2+Exp） |
  面向 85/kernelbench 评测：**g 也用 exp2**（A1/A2/B3 三组独立实现一致确认）。
  fla 仓 test_fwd_h.py 的 golden 是源码语义（g 用 exp）——两个 golden 不同，按评测目标选。
  v1 卡只写了源码语义（「g 自然对数域」），三组 agent 都需自行读 Model 仲裁——v2 显式化。
- **必须保留的语义点**（v1 已验证，全部保留）：
  1. g/gk 是 chunk 内**累积**量（cumsum；85 填充 = −U(0,0.08) 增量的 chunk 内 cumsum，Model:121-132）
  2. g_last/gk_last 取 **actual−1 行**（actual = min(CS, T−i·CS)）——⚠️ 取错静默错值
  3. **exp2 直译即可**：`tl.exp2 ≡ ×LN2+Exp`（实现侧延迟差 ≈0，A1 两态③闭环 [EXP]）——
     v1 的 ×LN2+Exp 逐位同构要求属不必要保守
  4. h[i] 存 chunk i 的**起始**状态（chunk0 = initial_state 或 0）
  5. kg 的预衰减只在调用方发生一次，kernel 内 k 不再乘 decay
  6. 中间域 fp32 + **三级舍入复刻**（85 Model 注释明示，精度墙关键）：states 存 k.dtype；
     C1 前 state 降 w.dtype 再 dot（fp32 累加）；C2 前 update_value 降 k.dtype 再累加
- **语义分层**：state_v_first 的 [K,V]↔[V,K] 由 wrapper permute-view 消化（§7）；
  output_final_state=false 的不可见 finalStateCompute 不需要（条带寄存器常驻）。

## 2. 原型与约束（同 v1，要点）

- 输入 k/w/u [B,T,H(K),K/V] fp16/bf16/fp32（85 套件 17/50 为 **fp32**，见 §5 注意）；
  g [B,T,HV]、gk [B,T,HV,K]；initial_state 可选；输出 states/v_new/final_state
- **85 套件给 gate_mode 显式参数且 g/gk 张量恒在**（gate_mode=0 都不用）——
  按 gate_mode 取 USE_G/USE_GK，勿按张量是否为 None 推断
- HV % H == 0（GQA，hk = hv // G）；vHeadDim ≤ 256
- **官方 50 case 的 K∈{7..80}、V∈{5..50} 非 2 的幂**：pow2/mask 泛化是编译前提（v1 修正保留）
- 变长：cu_seqlens int64 仅标量 load；wrapper 前缀和预计算（v1 保留）

## 3. 规格约束（同 v1）

910b / 910_93 / 950；数据 fp16/bf16 + state fp32 混合；tilingkey 不进 kernel；
源码 workspace 六段全部消解（条带常驻，结构性替代，GM 往返逐 chunk 可数）。

## 4. 分核策略（→ Triton grid）——v2 重写 [EXP]

- 任务轴：**(b, hv, v_block) 条带**（V 列条带独立，kernel .hpp:556,682），h 条带
  [BK, BV_TILE] fp32 常驻片内，chunk 维 program 内串行（v1 结论，三组实验一致采用）。
- **grid = NUM_TASKS（一块一 program）**——三个独立 [EXP] 证据：
  ① A1/A2（09-09）均一块一 program 起步直达 50/50；② MCTS 0901 冠军 7.2335x；
  ③ **v1 的 persistent-core 被 B3 证伪**：iter_0 即 13/50 UB 溢出，修复耗掉
  5 轮 gen 迭代（机制假设未验证：multibuffer 对外层领任务循环流水复制大块 tile +
  常驻 tile 活跃区间跨 task 延长）。C-011 已收窄：本族不适用 elementwise 家族默认。
- NUM_TASKS = B·HV·NVB 远不及 65535 上限（C-004）；无需运行时查核数。
- 变长：cu/co 由 wrapper 预计算（torch.cumsum），kernel 标量 load（O(1)）。

## 5. BLOCK / Tiling 参数——v2 重写 [EXP]

| 参数 | 规则 | 依据 |
|---|---|---|
| BT | = chunk_size（16/32/64/128 均 2 的幂） | v1 保留 |
| BK | max(16, next_pow2(K)) + kmask | tl.arange 2 的幂、tl.dot 最小维 16 |
| BV_TILE | 初值 max(16, min(64, next_pow2(V)))，然后 **守卫：BK·BV_TILE > 2048 元素则减半（至多到 16）**，NVB = cdiv(V, BV_TILE) | **A1 50-case 实测标定**：cap 4096/8192 反亏（latency-bound 下 program 数减少更慢，case50 200→229→312μs）；取代 v1 的解析 UB 分态（解析估计系统性偏低，hardware_platforms §3.2 校准警告） |
| GATE_MODE/USE_INITIAL/STORE_FINAL/SVF/VARLEN/IN_FP32 | constexpr | 编译期裁剪 |

- **fp32 输入（17/50 case）**：原生 `tl.dot(..., input_precision="ieee")`（A1/A2 双证
  可行；tf32 达不到容差）。**勿用 bf16x3 3-dot 仿真**——B3 误决策（fp32 组 0.80x vs
  A1 2.05x），其「原生 fp32 dot 不可用」结论只在其自造成的 UB 紧张上下文成立。
- **MCTS 大 shape（K=V=128、BT≤128）起点假设**（⚠️ 未实测）：BK=128 时 BV_TILE=16
  （2048 守卫自然落到 16）；B3 的 split-K/关 multibuffer 轨迹是大 shape UB 紧张的
  备选方案，仅当守卫规则仍溢出时再启用，勿作默认。

## 6. 片上内存预算——v2 原则

解析估算仅供 sanity check（实测占用系统性高于估计：dot staging/mask/cast 临时量未列，
hardware_platforms §3.2）。**以 §5 的 2048 元素实测守卫为准**；950（UB 248KB [SRC]）
与 910b（192KB）在 50-case 小 shape 下均富余。双缓冲由 multibuffer 默认承接
（bishengir 默认 true [SRC]；MCTS 探针 False −5.9%、B3 全局关净损失，双证勿关）。
**不配 num_stages/num_warps**（C-006）。

## 7. 流水编排与可达性——v2 更新

| Ascend C 手段 | Triton | 判定（依据） |
|---|---|---|
| C1→V1→C2→V2 跨核四段流水 / Catlass / Reg VF / direct-UB / zN / ComputeTail | 单 program 顺序 + 条带并行 | ❌机制/✅语义——结构性替代（v1 保留；**背景信息，勿尝试执行**） |
| k 的 ColumnMajor 读（C2 A 操作数） | 默认转置寻址 load [BK,BT]；**Phase 4 杠杆②做条件预转置** | ✅ [EXP 闭环]：A1 标定 NT≥3 或 (NT==2 且 CS≥32) 时 host `k.permute(0,2,3,1).contiguous()`（+21.1%，case50 −44%/case45 −59%）；NT=1/CS=16 反亏 ~10μs 固定拷贝，勿恒开 |
| exp2 | `tl.exp2` 直译 | ✅ [EXP] ≡ ×LN2+Exp（A1 两态③） |
| **state_v_first** | **wrapper permute-view 消化**（内部恒 [K,V] 连续写） | ✅ [EXP] +177%/+173% 双证（A1/B3）；对照法：kernel 内小 tile `tl.trans` 连续 store（A2：小 tile trans 比千级标量跨步 load 快 2.4-2.8x，bf16-svf1 组 3.84 vs 2.83 占优）——两法两态，勿回退为跨步散射 store（标量化慢 30~400 倍） |
| **tl.trans 禁令收窄** | 大 tile 禁（标量化 ~15x runbook）；小 tile（≤64×128）寄存器级可用 [EXP A2] | 按 tile 大小两态（tl_op_lowering.md 分态注） |
| **h 零初始化** | 全 False mask load 产零：`tl.load(..., mask=kv_mask & (kk[:,None]<0), other=0.0)` | ⚠️ **C-016**：`tl.zeros` 作 loop-carried dot 累加初值在 A1/A2 上下文触发 bishengir 段错误（无诊断；B3 幸存——上下文相关，不赌） |
| CV 同步 | sync_solver / auto_bind_sub_block 默认 true [SRC] | ⚠️ 编译器自动，不手写 |

## 8. Triton 骨架

完整骨架：`v2/triton_skeleton.py`（自包含，含 wrapper 与 85 语义 golden 复刻）。
结构一句话：一块一 program 领 (b,hv,v_block) 条带 → h 条带 fp32 初始化（全 False
mask load 产零或 initial_state）→ chunk 内循环｛store h[i] → w/u load → dot（fp32 走
ieee）→ v_new store → g/gk exp2 衰减（actual−1 行）→ k 转置寻址 load → dot 累加 h｝
→ 可选 final_state。关键纪律：BK/BV_TILE pow2+mask+2048 守卫；int64 标量大基址（4e，
MCTS 大 shape 必需）；无 `%`/num_stages/autotune；mask 全 int32 比较。

## 9. 精度标杆与验收（同 v1，一处替换）

- **标杆以评测目标为准**：85/kernelbench → `v2/triton_skeleton.py` 的
  `ref_chunk_gated_delta_rule_fwd_h`（复刻 85 Model 三级舍入 + 双 exp2 域）；
  fla 语义对照 → v1 骨架的 ref（g 用 exp）。
- 验收：rel diff 1e-4、≥99.99% 元素达标、单点 max_diff < 0.1，三输出分别判定。
- benchmark 双套件：正确性 = 官方 50 case；性能 = MCTS 8 大 shape（7.2335x 刻度）。

## 10. Agent actions——v2 重写

1. 取 `v2/triton_skeleton.py` 为起点（勿从零写）。类别判定：**状态常驻递推核**
   （operator_class_patterns §4）——grid 一块一 program，**勿 persistent-core**（§4）。
2. 语义以评测 golden 为准：g/gk 同 exp2/log2 域；g_last/gk_last 取 actual−1 行；
   三级舍入复刻（§1）。
3. **fp32 输入用原生 dot + `input_precision="ieee"`**；勿 bf16x3（§5）。
4. BK/BV_TILE 按 §5（pow2/mask + 2048 守卫）；UB 溢出先查守卫，勿走向 split-K /
   关 multibuffer 等重型改造（B3 教训：修复预算挤占两态预算）。
5. **Phase 4 待试杠杆清单**（全部 [EXP] 已证，按期望收益排序；skeleton 保持最小
   起点，以下由 Phase 4 逐一「假设→实测→采纳」）：
   | # | 杠杆 | 期望 | 实现要点 |
   |---|---|---|---|
   | ① | svf permute-view | +177%（已内建，确认勿回退） | states/fs 内部恒 [K,V] 连续，返回值 permute 视图 |
   | ② | k 条件预转置 | +21% | wrapper `if NT>=3 or (NT==2 and CS>=32): k = k.permute(0,2,3,1).contiguous()`；kernel 加 `K_PRE: tl.constexpr` 两态寻址（预转置后 tt 维 stride=1）；NT=1/CS=16 勿开 |
   | ③ | g 整块 load + 片上 select | +17%（gm=1 case） | g 按 [BT, HVP2] 连续块 load（hh = arange(0, next_pow2(HV))），`g_t = tl.sum(g_quad * (hh[None,:] == hv), 1)`、`g_last = tl.sum(tl.where(tt == actual-1, g_t, 0.0), 0)`；值逐位等价 |
   | ④ | h-无关 load 前置 | +4.5% | 循环体内 w/u/k/g/gk load 全部移至 states store 之前发射 |
   | ⑤ | svf 对照法 | 两态 | kernel 内小 tile `tl.trans(h)` 连续 store（A2 法；bf16-svf1 组更优） |
6. **勿重试清单**（[EXP]，operator_class_patterns §4.3 全表）：loop-carried tile 软件
   流水（scf.for memref 类型不匹配必崩，A1/B3 双命中）；k 侧门控移入 k_t（精度失败）；
   恒 k 预转置（小 case 反亏）；mask 全除；全局 multibuffer=False；UB cap 4096/8192；
   **失败上下文结论写成禁令**（C-012 变体：上下文失败只能标 ⚠️ 两态）。
7. 不要复现 AscendC 侧机制（§7 背景段）：GM workspace、CrossCoreFlag、Catlass、
   direct-UB、aclnn TransposeLastTwo。
8. 生成前过 preflight（backend_contracts 8 项 + dtype_support 4a~4e + **C-016**）。
9. 验收按 §9：先 50-case 正确性，再 MCTS 8 大 shape 性能。

## 11. 复现性声明——v2

v2 = v1（源码静态分析，confidence: inferred）+ 三组同机生成实验回流（confidence:
**measured**，910b、50-case 套件、profiler vs_torch 几何平均）：

| 组 | 知识条件 | geomean | 关键轨迹 |
|---|---|---|---|
| A1（09-09） | KB（含 A 卡 + runbooks） | **2.3535x** | gen 2 轮；Phase 4 0.5742→2.3535（杠杆①~④+⑤两态） |
| A2（09-09） | 仅 A 卡（KB 禁用） | 1.6873x | gen 3 轮；BV 扫描定 BV≤32；svf 小 tile trans 法 |
| B3（09-14） | v1 卡 + KB | 1.3251x | gen 5 轮（persistent UB 修复）→ 一块一 program；bf16x3 误决策致 fp32 组 0.80x |
| **B4（09-15）** | **v2 卡 + KB** | **2.3456x** | gen 3 轮（2 轮修 v2 骨架初版缺陷，见下）；Phase 4 严格按 §10 杠杆清单：②+32.7% ③+13.8% ④+5.4% 全部采纳（1.4674→2.3456），⑤实测拒绝；fp32 组 0.80→**1.75**（=A1）；opt_iter_0 探出 C-017（stride constexpr 误编译） |

**B4 对 v2 的验证结论**：v2 修订全部生效——无 persistent UB 溢出、无 bf16x3、
杠杆清单可移植（A1 资产在独立会话复现）；终值与 A1 持平（差 0.3%，噪声带内）。
**v2 初版骨架的两处缺陷**（已修，v2.1）：① w 的 head 维误写 hk（应为 hv，v1 原正确，
重写时回归——B4 iter_1 的 7 个 GQA case 全错）；② wrapper BV 守卫用 max/min/while，
触管线 AST checker 禁令（B4 iter_0）——两处共耗 B4 两轮 gen 迭代，**v2 骨架未过
端到端回放即交付的教训**：骨架改动（哪怕一行）也要按「不低于原论断证据强度」复核。

**v1 被证伪的两条建议**（v2 已修正）：① persistent-core grid（§4）；② 解析 UB 分态
（§5/§6 改实测守卫）。**v1 验证成立的内容**：条带任务轴、pow2/mask 泛化、语义六点
（golden 分叉除外）、int64 4e、svf wrapper 消化方向、禁 num_stages/multibuffer 默认。
分组数据与根因分解见 `generation_retrospective_20260915.md`。
**未实测**：950 平台；K>128 大 shape 的 2048 守卫表现；MCTS 套件 v2 复测。

## 维护记录

- 2026-09-09 v1 初版（旧流程）；2026-09-14 v1 新流程重提取（两处修正后被实验部分证伪）。
- 2026-09-15 **v2**：三组生成实验（A1 2.3535x / A2 1.6873x / B3 1.3251x）回流——
  grid 回一块一 program；fp32 原生 ieee dot；BV 2048 元素实测守卫；语义明确面向评测
  golden（g/gk 双 exp2）；exp2 直译；C-016 零初始化写法；新增 Phase 4 杠杆清单与
  勿重试清单；❌ 机制项归入背景段明示勿执行。
- 2026-09-15 **v2.1 + B4 验证**：B4 复跑（v2 卡+KB）2.3456x ≈ A1，v2 修订全部生效；
  修复骨架初版两处缺陷（w 索引 hk→hv；BV 守卫改纯三元表达式，防管线 AST checker）；
  §11 增 B4 行；opt_iter_0 新陷阱回流为 backend_contracts C-017。
