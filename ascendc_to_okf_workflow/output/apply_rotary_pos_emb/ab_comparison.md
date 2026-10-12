# A/B 对比报告 — apply_rotary_pos_emb 新旧流程重提取

> 日期：2026-09-14 ｜ A = 2026-09-08 旧流程（无 backend_facts，baseline_20260909/）
> B = 2026-09-14 新流程（强制检索 backend_facts，六任务纪律）
> 性质说明：本算子是 backend_facts 部分契约的**来源**（C-001/C-002/C-006/C-007/C-012
> 由其 P4 蒸馏），因此 B 组骨架与 A 组 P4 收敛版同构是预期自洽，不构成独立验证。
> 独立验证价值在 §3 的反事实分析：**若 backend_facts 先于 P4 存在，哪些迭代可省**。

## 1. 产物对比总览

| 维度 | A（旧流程） | B（新流程） |
|---|---|---|
| P0/P1 | 高质量，锚点齐全 | 复用 + 抽验 4/4 锚点有效（新规程 P0/P1 不查外部库，行为不变） |
| P2 后端论断证据 | 无证据标注，混模型先验 | 六任务检索全命中首层，逐条 [EXP]/[DOC]/[SRC] |
| P2 错误论断 | 3 处（§2 表） | 0 处（已修正） |
| 骨架 preflight | 流程中无此动作 | 12 项 grep，0 真命中 |
| 卡片外部事实 | 复写在卡内（UB 192KB 等） | 只挂 backend_facts 引用，不稀释 |

## 2. B 组在 P2 阶段即拦截的 A 组错误（无需 P4 即可发现）

| # | A 组旧版论断 | B 组拦截点 | A 组实际付出的代价 |
|---|---|---|---|
| 1 | mapping §4「vs UB 192KB/核」 | 检索任务④：hardware_platforms [SRC] 950=248KB | 预算偏保守（方向安全，未致错） |
| 2 | mapping §3/§5「num_stages autotune {1,2,3}」 | 检索任务③⑥：C-006 运行时不生效 [EXP] + compiler_options multibuffer 默认 true [SRC] | P4 KB 组 autotune 残留致 34/45 case 劣化，多花迭代清算 |
| 3 | mapping §5「strided-2 为默认、split/interleave 为变体、实测择优」 | 检索任务②：strided2 契约 ❌ 禁用 [EXP]（3 类 bug） | P4 撞静默错值 + 相邻行污染缺陷后才改连续+重排 |
| 4 | mapping §5「cos/sin 自然驻留 L1/L2、勿复刻」 | 检索任务②：C-012 等价断言须实测；默认设计即组级外提 | P4 实测推翻后 +100%（2.0956→4.2060），断言曾压住优化方向 |

## 3. 反事实分析：backend_facts 若先于 P4 存在

A 组 P4 报告的 12 项缺陷审计（baseline validation_report §5）中，**7 项在 B 组 P2 阶段
即被 backend_facts/preflight 覆盖**（#3 strided-2、#6 num_stages、#7 mask/`%`/stride-0、
#8 组外提、#2 等价断言、#1 骨架静态自检进 P2 质量门、#5 trans 标量化入 dtype 检索面）；
**2 项属流程规范**（#9 KB 自包含、#11 材料漏扫）已由 README 修订覆盖；
**3 项仍需实测**（#4 shifted-load 劣化、#10 Q/K 融合裁决、#12 官方容差锚点）——
这些是 P4 不可替代的部分，符合「backend_facts 压缩迭代、不消灭实测」的定位。

量化估计（基于 A 组实验记录）：B 组预期省去 strided-2 踩坑（≥1 轮）+ autotune 清算
（≥1 轮）+ 外提方向纠正（≥1 轮）≈ **P4 迭代轮数减半**；起点质量（baseline kernel）
从 A 组的 2.0956 直接起步于 ~4.2 结构。

## 4. B 组新暴露的问题（回流项）

1. **preflight grep 需排除注释/docstring**：首轮脚本把头注中的禁令文字误报为命中
   （`%`、`num_stages`、`autotune` 各 1~2 假阳性）——已改「剥离注释后扫描」，
   建议把该实现固化进 backend_contracts.md 文末清单的使用说明。
2. **wrapper 层 `%` 合法**：C-001 适用范围是 @triton.jit kernel 体；wrapper（host Python）
   的 `RD % 2` 不违禁——契约文字已含「kernel 内」，preflight 实现须按函数边界切分。
3. **`_vector_core_num` fallback=40 是 950 风险点**：950 向量核 8~72，fallback 40 在
   查询失败时 silently 错配 grid——已在骨架注释标警告；建议硬件查询失败时 raise 而非兜底
   （待 P4 裁决，暂不写死）。
4. **tiling UT UB mock=196608B 与 248KB 规格并存**：UT 是 mock 非硬件值，卡片引用时
   须注明来源性质——已写入 manifest 头注与卡片 §6。

## 5. 结论

- 对**本算子**（backend_facts 的来源算子）：新流程产物与 P4 收敛版自洽，证据链从
  「实测孤证」升级为「[SRC] 源码 + [EXP] 实测 + [DOC] 文档」三锚互证，卡片可维护性
  （引用而非复写）显著改善。
- 对**流程验证**：六任务检索纪律在真实算子上行得通，首层命中率 6/6；preflight 12 项
  可作为 P4 前置硬门。下一轮 A/B 应在 **backend_facts 的非来源算子**（如 mhc 剩余 7 个）
  上重做，才能获得独立的「命中率/迭代压缩」数据。
