# 「已知实例」append 与消费反馈回链规程

> 适用：S3（无提升/禁用记录 → 优化点条目）、S2（修复闭环 → CT/AP/DG 条目）、S4（消费反馈 → 被消费卡）。
> 原则出处：ingest 编排器 §3.4「一概念一 canonical 卡」+「干净上下文合并」。

## 一、判断 append 还是新建

1. 读目标 `runbooks/<category>/index.md` + `grep -r <主题关键词>` 全文确认。
2. 判据（由**干净上下文子 Agent**判定，主 Agent 不得在读完单算子细节后自行合并）：
   - 既有条目的**机制**与本候选相同（同为 load 重排 / 同为双缓冲 / 同为对齐约束）→ **append**
   - 机制不同但维度相近 → 新增条目（新 OPT/AP/CT 编号，ID 取该目录当前最大 +1，不复用不重排）
   - 机制相同但**矛盾**（本例实测与既有条目结论相反）→ 不覆盖，按「矛盾绝不静默覆盖」处理：
     在既有条目 append 矛盾实例 + `> ⚠️ 矛盾标记（<日期>，<证据路径>）`，并进 `log/` 待人工裁决

## 二、runbook 条目 append 格式

在既有条目的「已知实例」小节追加一行（与既有风格一致）：

```
**已知实例**：<op 或算子类别> §<锚点>（<ops 卡相对路径#锚点>）
```

带数据的增强写法（S3 无提升记录推荐）：

```
- <算子类别>（<arch>）：<一句话条件与结果>，<speedup a→b>（证据：<json 字段路径>）
```

示例：

```
**已知实例**：
- chunk 递推+CV 融合类（ascend910b1）：循环内依赖就绪 load 提前发射无提升，
  22.0006 vs 22.2468（fw 侧 profiling 噪声同量级）（证据：op_94_94_Gdn2Fwd_20260914_1219_8124/output/opt_iter_0/optimized_perf_result.json.speedup_vs_torch）
```

**同时**：对应 ops 案例卡（若 S1 产出）须在本卡方向上有反向链接（「实例化 ⇄ 已知实例」双向），ops 卡不存在时（S1 未触发）条目内写清算子名 + 工作目录证据路径即可。

## 三、S4 消费反馈回链

回链对象：manifest `consumed_cards` 中 `relevance_level == strong` 且 Agent 逐张确认**实际影响了本次设计/修复/优化决策**的卡。

在被消费卡正文合适小节（既有「已知实例」或相关主题小节）append：

```
- <op-name>（<category>，<arch>，<YYYY-MM-DD>）：<一句话使用效果，如"其依赖二分框架直接被本次
  双骨架 dispatch 复用">（<ops 卡相对路径#锚点 或 工作目录 report.md 路径>）
```

纪律：
- **宁缺毋滥**：弱命中（weak/medium）与"读了但没用上"的卡不回链，防反馈膨胀稀释信号
- ops 案例卡 → 被消费 runbook 卡的**正向链接**由案例卡承担；本节只做**反向**补链
- 被消费卡是 ops 卡（非 runbook）时，回链到其「复用决策要点」小节

## 四、append 后必做

1. 更新该条目 frontmatter `updated_at`
2. 对应分类 `index.md` 若有"新增条目"叙事之外的实例计数/描述变化，同步一行
3. `log/<YYYY-MM-DD>.md` 记：append 对象、来源工作目录、证据指针
4. 图谱增量：`okf_graph.py candidates → judge → inject → viz → verify`（append 也可能产生新边）
