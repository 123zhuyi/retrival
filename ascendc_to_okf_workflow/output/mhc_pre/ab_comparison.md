# A/B 对比报告 — mhc_pre 新旧流程重提取

> 日期：2026-09-14 ｜ A = 2026-09-03 旧流程（无 backend_facts，baseline_20260903/）
> B = 2026-09-14 新流程（强制检索 backend_facts，六任务纪律）
> **性质说明：mhc_pre 不是 backend_facts 任何契约的来源算子**（契约蒸馏自 add /
> fused_bias_leaky_relu / apply_rotary_pos_emb 的 P4 回放）。因此本次 B 组对 A 组的
> 拦截构成**独立验证**，区别于 apply_rotary_pos_emb 的自洽性演示。

## 1. 产物对比总览

| 维度 | A（旧流程，9-03） | B（新流程，9-14） |
|---|---|---|
| P0/P1 | 高质量，锚点齐全 | 复用 + 抽验 5/5 锚点有效 |
| P2 后端论断证据 | 无证据标注，混模型先验 | 六任务检索全命中首层，逐条 [EXP]/[DOC]/[SRC] |
| P2 错误/潜伏缺陷 | 2 处错误 + **1 处潜伏 bug** | 0 处（全部修正，§2） |
| 骨架 preflight | 流程中无此动作 | 12 项 grep，0 真命中 |
| 转化主结论 | 单 kernel + 顺序 K 循环（正确） | 维持不变（B 组确认其结构性论证成立） |

## 2. B 组在 P2 阶段拦截的 A 组缺陷（无需 P4 即可发现）

| # | A 组旧版论断 | B 组拦截点 | 若不拦截的后果 |
|---|---|---|---|
| 1 | mapping §4「~110KB ≤ 192KB UB ✅」 | 检索任务④：hardware_platforms [SRC] 950=248KB | 预算偏保守（方向安全）；n=32 大 tile 场景的切分上限判断基准错误 |
| 2 | 骨架 `get_device_properties(...)["num_vectorcore"]`；mapping §1「Mix 场景无需区分 AIC/AIV 角色」 | 检索任务⑥：C-011/operator_class_patterns——CV 融合 grid=num_aicore；「无需区分」是无实测等价断言，违反 C-012 | grid 错配：950 上 num_vectorcore=2×num_aicore，**program 数翻倍、每核任务量减半**，且与编译器 CV 调度假设不符——性能静默劣化，P4 才能察觉 |
| 3 | 骨架寻址 `offs_m[:, None] * K`（int32 张量） | 检索任务⑤：dtype_support i64 规则 → 反向审计 benchmark shape：**T·K ≥ 2³¹ 时 int32 偏移溢出，13 case 中 6 个触发**（[16,16384,4,4096]/[8,16384,8,4096] 达 4.3e9） | **静默错值**：地址回绕读错数据，无报错；大 shape case 全挂，且 grid=1 小 shape 调试不复现——属最难排查的一类缺陷 |

**第 3 条是本次重提取的最高价值产出**：A 组骨架直接交付会给 agent 埋一个
「小 shape 通过、benchmark 大 shape 全挂」的潜伏 bug；B 组在 P2 静态阶段即拦截，
修复方案（int64 标量基址 + int32 块内偏移）同时满足 i64 标量化规避纪律
（int64 不进向量 lane）。

## 3. 六任务检索命中记录

全部命中 backend_facts 首层（6/6），未下沉 zh/IR 原文。关键首层证据：
tl.dot=T2 Cube（tl_op_lowering）；CV 融合选项默认值全 true/2（compiler_options [SRC
Options.td]）；950 UB=248KB（hardware_platforms [SRC]）；950 无 bf16 向量指令、
i32 EQ 比较向量安全（dtype_support [SRC]）；CV 融合 grid=num_aicore（C-011）。

## 4. A 组做对的、B 组确认维持的决策

- 三 tiling 模式 → 单 kernel + 单 program 顺序 K 循环（B 组复核：该「丢弃」是结构性
  等价替代——无跨核写同一张量，论证完备，不违反 C-012）
- phi 拆三段三次 tl.dot、alpha/bias wrapper 预展开、fp32 计算域、段布局五条语义点
- autotune 保留的适用性判断（本 kernel 非 µs 级，C-007 不适用；B 组补充 C-014 约束
  与「交付前固化」要求）
- benchmark 定义域超集警告（n 至 32 不硬编码 Ascend C n≤8 约束）

## 5. B 组新暴露的问题（回流项）

1. **int64 寻址陷阱应升格为通用契约**：本轮通过「dtype 检索 → 反向审计 shape」
   偶然捕获；建议 backend_contracts/dtype_support 增加显式条目「numel ≥ 2³¹ 的
   张量寻址：int64 仅限标量基址，块内偏移 int32」——所有大 K/大 T 算子（mhc 全族、
   ChunkGatedDeltaRule 等）通用。（待回流）
2. **preflight 清单可增一项**：`offs_*[:, None] * K` 类纯 int32 行寻址模式的静态
   检查（需 shape 上限信息，半自动）。
3. `_aicore_num` fallback=24 与 apply_rotary 的 `_vector_core_num` fallback=40
   同属「查询失败 silently 错配」风险——两算子一致建议：查询失败应 raise（待 P4 裁决）。

## 6. 结论

- 对**流程验证**（首次独立验证）：backend_facts 在非来源算子上 6/6 首层命中，
  拦截 2 处错误论断 + 1 处 P4 级潜伏 bug（int32 寻址溢出）——该 bug 若在 agent
  生成阶段爆发，预估消耗 ≥2 轮 run-fix（小 shape 复现不了 → 怀疑 evaluator →
  逐 case 二分），且容易误判为精度问题。
- 旧流程主结论（单 kernel 结构性简化）经新流程复核**成立**，说明 A/B 差异不在
  转化方向而在**后端事实的准确性与潜伏陷阱的静态拦截**。
- 本卡包（B 组）与 `baseline_20260903/`（A 组）可直接用于后续 agent 生成效果的
  双盲对比实验。
