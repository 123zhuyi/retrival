---
id: doc-triton-ascend-zh-triton-api-triton-index
title: triton
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/triton/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/triton/index.md
content_sha256: 55c8dbe65a29b592fdde45a475b4e6a9bbe824b829eb536fa7522f4e8edf2b9b
snapshot_body_sha256: 55c8dbe65a29b592fdde45a475b4e6a9bbe824b829eb536fa7522f4e8edf2b9b
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
---

# triton

| API | 简要说明 |
|-----|----------|
| [jit](./jit.md) | JIT 装饰器 - 使用 Triton 编译器编译函数 |
| [autotune](./autotune.md) | 用于自动调优一个经过 `triton.jit` 编译的函数的装饰器 |
| [heuristics](./heuristics.md) | 用于指定如何计算某些元参数值的装饰器 |
| [Config](./Config.md) | 一个表示自动调优器可以尝试的可能内核配置的对象 |

```{toctree}
:maxdepth: 3
:hidden:

jit.md
autotune.md
heuristics.md
Config.md
```
