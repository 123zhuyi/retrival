# 新 source 接入 pipeline（路由 1）

首次把某个上游（新轨迹集 / 新文档仓 / 新算子批次）接入 triton-ascend 知识库时走本流程。

## 0. 前置确认

- 明确 source 类型：**轨迹集**（Agent 探索产物，如 triton_ascend_output 的 report/summary/session）还是**文档仓**（triton-ascend 官方文档，`<上游文档仓占位>`）。
- 明确知识归属分类（1–6，见编排器 §6）；跨分类材料按主题拆分，分别路由。
- 在内容根 `index.md` frontmatter 记录 pin（轨迹集 `trajectory_repo/ref/commit`；文档仓 `upstream_repo/ref/commit`），即 watermark 起点。

## 1. 盘点（harvest）

- 轨迹集：列出全部有效案例（`summary.json` 中 `success: true` 且 `passed_cases == total_cases` 的探索），建立 `inventory`：算子名、轨迹路径、最终 speedup、命中的优化点、失败迭代记录。
- 文档仓：列出全部文档，按概念归堆（候选卡种子）。
- 无效/失败轨迹不丢弃：其中"诊断 → 修改 → 仍失败"的记录是 **AP-*（反模式）/ CT-*（约束陷阱）** 的原料，单独登记。

## 2. 判定（curate·判定）

逐案例/逐文档决定：

1. **是否进库**：对"设计/优化 Triton-Ascend 算子的模型"有知识增量才进；纯环境噪声 skip（记理由）。
2. **进哪棵树**：算子特定事实 → `ops/` 案例卡；跨算子可复用规律 → `runbooks/<category>/` 条目；官方事实 → `reference/`。
3. **进哪个分类**：按设计文档 18 个二级维度判定，写 `dimension` 字段；判不准时停下问人，不猜。
4. **新建 or 更新**：先 `triton-knowledge-query` 检索已有卡；同概念命中 → update（append 已知实例），否则 new。
5. 记 ledger（决策账）：`{item, action, card_id, category, dimension, reason}`。

## 3. 著作（curate·著作）

- 委派对应分类的生产 skill 执行（见各 skill SKILL.md 的五阶段工作流）。
- 蒸馏非照搬；runbook 条目算子无关（占位名骨架）；性能数字必须指回轨迹 benchmark 记录。
- GPU 概念先转换为 AICore/Vector/Cube/Mask/Repeat 语境再入库。

## 4. 收尾（finalize，编排者集中写）

1. 逐层 `index.md`（每级只列本层直接子项）。
2. 检索重建：`triton-knowledge-query` 的 `knowledge_query.py build`。
3. 图谱增量：`okf_graph.py candidates → judge → inject → viz → verify`。
4. 覆盖账：`union(所有卡 sources) ∪ skipped(带理由) == 本 source 全部条目`。
5. verify：`knowledge_query.py verify` + `okf_graph.py verify` + `knowledge_lint.py`。
6. 日志：`log/<date>.md` 顶部插入 `## [HH:MM] sync | <source> 新接入`（Summary/Details + Changes + References；skip 理由入 Details）。
7. 推进 watermark（pin 写入内容根 index.md frontmatter）。
