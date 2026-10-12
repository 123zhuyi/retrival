# A/B 对比报告 — chunk_gated_delta_rule_fwd_h 新旧流程重提取

> 日期：2026-09-14 ｜ A = 2026-09-09 旧流程（无 backend_facts，baseline_20260909/）
> B = 2026-09-14 新流程（强制检索 backend_facts，六任务纪律）
> **性质说明**：本算子是三方独立——① backend_facts 契约不源自本算子；② 历史 MCTS
> 生成实验（9-01）早于旧卡（9-09）与 backend_facts（9-11）；③ 旧卡从未 P4 回放。
> B 组对 A 组的拦截构成独立验证，且 MCTS 的 [EXP] 数据可反向交叉验证 B 组结论。

## 1. 产物对比总览

| 维度 | A（旧流程，9-09） | B（新流程，9-14） |
|---|---|---|
| P0/P1 | 高质量，锚点齐全 | 复用 + 抽验 6/6 锚点有效；**修正 P1 两处**（UB 常数无锚点误述；regbase 快照表述过宽） |
| P2 后端论断证据 | 部分引用 runbook（trans/num_stages 已知），其余无证据 | 六任务检索全命中首层，逐条 [EXP]/[DOC]/[SRC] |
| P2 错误/潜伏缺陷 | 2 处错误论断 + **1 处骨架潜伏缺陷** | 0 处（全部修正，§2） |
| 骨架 preflight | 流程中无此动作 | 13 项 grep，0 真命中 |
| 转化主结论 | (b,hv,v_block) 条带常驻（正确） | 维持不变（B 组确认结构性论证成立） |

## 2. B 组在 P2 阶段拦截的 A 组缺陷（无需 P4 即可发现）

| # | A 组旧版论断 | B 组拦截点 | 若不拦截的后果 |
|---|---|---|---|
| 1 | extraction §5「UB = 192KB（Arch::Ascend950 资源池）」；卡 §6「峰值贴 192KB 上限，BT=128 建议降 BV=64」 | 检索任务④：hardware_platforms [SRC] **950=248KB**；192KB 是 A2/910b 值 | 「降 BV=64」被误当通用结论——实为 910b 分态（MCTS 910b1 冠军正是 BV=64 [EXP]，互证）；950 上 BV=128（~190KB vs 248KB）本可行，**并行条带数被人为翻倍、串行递推长度不变、性能静默劣化** |
| 2 | 卡 §4「一块一 program 即可……不属于 elementwise 家族的超额 block 负引导场景」 | 检索任务⑥：C-011 persistent-core 范式 + C-012（自我豁免 = 无实测等价断言） | grid=NUM_TASKS 在海量条带（varlen 大 B）时超额 launch；与 CV 融合算子 grid=num_aicore 的契约不符。B 组改 persistent-core（对长串行任务零成本），旧形态留作 P4 两态 |
| 3 | 骨架 `tl.arange(0, K)` / wrapper `K=K` 原样传入 | 检索任务⑤ dtype 审计 + benchmark case 实测 shape：**官方 50 case 的 K∈{7..80}、V∈{5..50} 非 2 的幂** | **旧骨架在官方正确性套件上编译即失败**（tl.arange 须 2 的幂；tl.dot 最小维 16）——agent 拿旧卡生成，第一轮就撞编译错误，且卡片文字（§5 表格写 next_pow2）与骨架实现不一致，排查成本高 |

**第 3 条是本次重提取的最高价值产出**：A 组卡片的文字与骨架自相矛盾（表格写了
next_pow2，代码没做），只有 MCTS 的 K=V=128 大 shape 能掩盖。B 组以
BK=max(16,next_pow2(K)) / BV_TILE=max(16,min(128,next_pow2(V))) + 全程 mask 修复，
官方 50-case 正确性套件与 MCTS 8 大 shape 性能套件一套代码通吃。

## 3. 六任务检索命中记录

全部命中 backend_facts 首层（6/6）。亮点：**任务③获得本算子独立 [EXP] 交叉验证**——
compiler_options.yaml 的 multibuffer 默认 true [SRC] 与 MCTS opt-round-5 探针
（multibuffer=False → 6.8053x vs True 7.2335x，-5.9% 回退）互相印证；任务④的
910b BV=64 分态与 MCTS 冠军配置（BV=min(64,V)，round-2 实测 8 shape 全胜）吻合。

## 4. A 组做对的、B 组确认维持的决策

- (b,hv,v_block) 状态条带常驻并行轴（V 列条带独立是关键结构观察）；GM workspace 六段、
  CrossCoreFlag、direct-UB、ComputeTail、zN 的勿复刻清单（B 组复核：均为结构性替代，
  GM 往返次数可数论证，非 C-012 断言）
- 禁 tl.trans 用转置寻址 load、禁 num_stages/autotune、禁 `%`、int32 mask
  （A 组 9-09 已吸收 9-08/09 README 增补的 P4 教训——这部分不是新流程独有增量）
- exp2 = ×LN2+Exp 逐位同构、tl.exp2 两态未实测（C-012 合规写法 A 组已具备）
- g 的 chunk 内 cumsum 语义、actual−1 行、填充纪律

## 5. B 组新暴露的问题（回流项）

1. **「卡片文字与骨架实现不一致」是旧流程一类隐患**（本次 §0.6-3：表格 next_pow2 vs
   代码直传 K）——新流程的「骨架 preflight 进 P2 质量门」可静态拦截此类不一致，
   建议质量门增加一条「骨架输入域 ≠ 卡片声明输入域时报警」（需 benchmark shape 信息，
   半自动）。（待回流工作流 README）
2. **P1 也可能含无锚点事实错**（本次 UB 192KB 误述无源码锚点）——复验规程的锚点
   抽验应优先抽「数字型论断」（容量/核数/对齐），这类错误会沿 P1→P2→卡片放大。
   （待回流工作流 README P1 质量门）
3. MCTS 实验的 BV=64/multibuffer 结论已反向汇入本卡 §5/§6 作为 910b 分态依据；
   其 `operator_experience.md` 的其余经验（GQA 核内映射、varlen 前缀和）与本卡
   决策一致，无需额外回流。

## 6. 结论

- 对**流程验证**（第二次独立验证，继 mhc_pre 之后）：backend_facts 在非来源算子上
  6/6 首层命中，拦截 2 处错误论断 + 1 处**编译级**潜伏缺陷——第 3 条意味着 A 组
  卡包直接交付时 agent 在官方正确性套件上**第一轮必败**，B 组在 P2 静态阶段归零。
- 与 mhc_pre 的拦截类型互补：mhc_pre 是「运行期静默错值」（int32 寻址溢出，最难排查），
  本算子是「编译期即败 + 性能分态误判」（pow2/mask + UB 平台常数）——两类都验证了
  「六任务检索 + 骨架 preflight 进质量门」的静态拦截价值。
- MCTS 历史实验为 B 组两个结论（910b BV=64、multibuffer=True）提供了独立 [EXP]
  互证——三层体系（backend_facts 蒸馏 + 源码仲裁 + 生成实验回放）首次闭环。
- 本卡包（B 组）与 `baseline_20260909/`（A 组）可直接用于后续 agent 生成效果的
  双盲对比实验；benchmark 双套件（官方 50 case 正确性 + MCTS 8 大 shape 性能）现成。

## 附：B 组卡包生成实验已完成（2026-09-15 补记，同日二修）

用户用 B 组卡 + 骨架执行了生成实验，且 cannbot-skills 存档中存在同算子的完整三组
历史实验（同机 910b、同 50-case 套件、同指标）：

| 组 | 知识条件 | geomean |
|---|---|---|
| A1（09-09） | KB（内含 A 卡 + runbooks） | **2.3535x** |
| A2（09-09） | 仅 A 卡（KB 禁用） | **1.6873x** |
| B3（09-14） | B 卡 + KB | **1.3251x** |

**对本文 §2 结论的修正**：三处「拦截」中，①（UB 平台常数）与 ③（非 2 幂编译失败）
仍成立；**②（grid 改 persistent-core）被实验证伪——A 卡的「一块一 program」原本
正确，B 组按 C-011 的「修正」是回归**（B3 iter_0 即 13 个 UB 溢出，5 轮 gen 迭代
大多耗在爬出该兔子洞）。差距定量分解（fp32 的 bf16x3 误决策 ≈22.5% + A1 独有四
杠杆 ≈36.2%）与全部根因分析见 `generation_retrospective_20260915.md`（三组对比
完整版）。教训已回流：C-011 按算子类收窄、`operator_class_patterns.md` 新增
「状态常驻递推核」范式、对旧卡论断的「修正」须要求不低于原论断的证据强度。
