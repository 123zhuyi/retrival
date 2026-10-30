---
id: doc-triton-ascend-zh-triton-api-random-number-generation-index
title: 随机数生成
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Random_Number_Generation/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Random_Number_Generation/index.md
content_sha256: b1bc9e48915f39a91bbba8e7cc6d10c15323092feb609b1893b4c0cdcfb80b16
snapshot_body_sha256: b1bc9e48915f39a91bbba8e7cc6d10c15323092feb609b1893b4c0cdcfb80b16
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- random
---

# 随机数生成

|api|简要说明|
|--|--|
|[randint4x](./randint4x.md) |给定 1 个seed 标量和 1 个offset 块，返回 4 个 int32 类型的随机块 |
|[randint](./randint.md) |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个 int32 类型的随机块 |
|[rand](./rand.md)   |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个在 U(0,1) 中的 float32 类型的随机块 |
|[randn](./randn.md)   |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个在 N(0,1) 中的 float32 类型的随机块 |

```{toctree}
:maxdepth: 3
:hidden:
randint4x.md
randint.md
rand.md
randn.md
