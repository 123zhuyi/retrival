# 4 窗口并行打磨任务总览

## 当前状态

- 303 个 raw YAML 经验已全部转换为 OKF runbook 卡片并写入目标库 `triton-ascend-kb.bak.20260807/runbooks/`（exp_*.md）
- ~170 条已合并挂载到 20 个已有 runbook 卡
- 质量缺口：274 卡「坏实践=待补充」、303 卡「失效边界=待补充」、56 卡 confidence=provisional、部分挂载可能主题漂移

## 分工

| 窗口 | 任务书 | 范围 | 依赖 |
|---|---|---|---|
| A | `TASK-A-compute-pipeline.md` | `runbooks/compute-pipeline/exp_*.md`（148） | 无 |
| B | `TASK-B-memory-sram.md` | `runbooks/memory-sram/exp_*.md`（108） | 无 |
| C | `TASK-C-other-categories.md` | portability(33)/meta-skills(8)/algorithm-numerics(4)/autotuning(2) 的 exp_*.md | 无 |
| D | `TASK-D-merge-review-finalize.md` | 挂载 review（已有卡） + 索引/图谱/lint 收尾 | 收尾部分等 A/B/C |

## 公共纪律（4 个窗口都要遵守）

1. 只写自己范围的文件，不越界
2. 不改 frontmatter 的 `created_at`；confidence 校准归窗口 D
3. 不增删文件、不改文件名
4. 不 commit、不 push（用户统一处理）
5. 不跑 `okf_graph.py` / `knowledge_query.py` / `knowledge_lint.py`（仅窗口 D 在收尾时跑）

## 关键路径

- 目标库：`/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807/`
- 原始 YAML：`/home/w00836812/shixi_test02/cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/raw/experiences/`
- 规范：`/home/w00836812/shixi_test02/zmm_knowledge/Triton-Ascend-KernelWiki/skills/triton-knowledge-*-ingest/STRUCTURE-runbook.md`
- 批量合并报告（供参考）：`/tmp/merge_full_report.json`
