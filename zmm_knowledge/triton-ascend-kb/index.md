---
kind: index
type: bundle_index
title: triton-ascend 知识库
description: Triton-Ascend 算子开发知识库。包含 reference 开发指南、ops 算子实践案例、runbooks 优化经验，以及从 triton-knowledge-retrieval 迁移的 experiences 与 extracted kernels 反链。
---

# triton-ascend 知识库

> 本知识库由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/` 迁移并结构化而来，采用 OKF v1 schema 与 ingestion pipeline 规范。

## 内容树

- [reference/](reference/index.md) — triton-ascend API/文档蒸馏卡、op-class 开发指南、skeleton、SPEC
- [ops/](ops/index.md) — 算子案例卡（成功探索 → 设计 wiki）
- [runbooks/](runbooks/index.md) — 6 大分类优化经验库（核心）

## 跨树共享层

- `graph/` — 知识图谱产物（`edge_judgments.json` 判定缓存 + `viz*.html`）
- `search/` — 检索索引（`knowledge_query.py build` 产物）
- `log/` — 按天审计日志 +  legacy-retrieval-cache 归档
- `metadata/` — `type_map.yaml` / `tag_registry.yaml` / `aliases.yaml`

## 迁移统计

| 阶段 | 产出 |
|---|---|
| P0 预检 | 303 YAML experiences、268 extracted kernels 清点 |
| P1 reference | 23 张 reference 卡 + 40 个 legacy cache 归档 |
| P2 runbook | 25 张新 runbook 卡 |
| P3 反链挂载 | 303 experiences + 268 kernels 挂载到 37 张目标卡 |
| P4 闭环 lint | `kb_finalize.py` 全绿：0 blocker / 0 warn |

## 规范

`SPEC-*.md`：SPEC-Reference / SPEC-Source / SPEC-Graph / SPEC-Retrieve / SPEC-Knowledge-Compile / SPEC-Version-update。
