---
kind: index
type: bundle_index
title: triton-ascend 知识库（骨架占位）
description: Triton-Ascend 算子开发知识库根导航。本骨架仅展示目录结构，真实知识正文后续由摄入流程填充。
---

# triton-ascend 知识库（骨架占位）

> 本目录是 `plugins-community/triton-ascend-knowledge/examples/knowledge-base-skeleton/` 的骨架副本。
> 初始化新知识库时复制本目录；真实知识由 `triton-knowledge-ingest` 编排的摄入流程填充。

## 内容树

- [reference/](reference/index.md) — triton-ascend API/文档蒸馏卡（`<上游文档仓占位>`）
- [ops/](ops/index.md) — 算子案例卡（成功探索 → 设计 wiki）
- [runbooks/](runbooks/index.md) — 6 大分类优化点库（核心）

## 跨树共享层

- `graph/` — 知识图谱产物（`edge_judgments.json` 判定缓存 + `viz*.html`）
- `search/` — 检索索引（`knowledge_query.py build` 产物）
- `log/` — 按天审计日志 `<YYYY-MM-DD>.md`（倒序）
- `metadata/` — `type_map.yaml` / `tag_registry.yaml` / `aliases.yaml`（占位）

## 规范

`SPEC-*.md`（占位，后续从 Asc 规范改写）：SPEC-Reference / SPEC-Source / SPEC-Graph / SPEC-Retrieve / SPEC-Knowledge-Compile / SPEC-Version-update。
