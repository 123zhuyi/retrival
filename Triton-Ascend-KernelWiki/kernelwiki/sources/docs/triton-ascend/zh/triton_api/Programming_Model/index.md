---
id: doc-triton-ascend-zh-triton-api-programming-model-index
title: 编程模型
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Programming_Model/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Programming_Model/index.md
content_sha256: e04f177608da3d8d14fe554bd5210b32c5293da14527c444451dd54da1358c95
snapshot_body_sha256: e04f177608da3d8d14fe554bd5210b32c5293da14527c444451dd54da1358c95
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- control
---

# 编程模型

|api|简要说明|
|--|--|
| tensor | 表示一个值或指针的 N 维数组 |
| [program_id](./program_id.md) | 沿指定轴返回当前程序实例的 id |
| [num_programs](./num_programs.md) | 沿指定轴返回当前程序实例的数量 |

```{toctree}
:maxdepth: 3
:hidden:
program_id.md
num_programs.md
