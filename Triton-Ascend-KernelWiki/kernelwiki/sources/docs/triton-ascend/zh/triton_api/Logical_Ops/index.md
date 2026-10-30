---
id: doc-triton-ascend-zh-triton-api-logical-ops-index
title: 逻辑操作
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Logical_Ops/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Logical_Ops/index.md
content_sha256: 92a37b876b7faf34973e08953f3683a0ecfbeb32533527bbc6cb793321d43a78
snapshot_body_sha256: 92a37b876b7faf34973e08953f3683a0ecfbeb32533527bbc6cb793321d43a78
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- math
---

# 逻辑操作

|api|简要说明|
|--|--|
|[and](./and.md) | 逻辑与操作 |
|[or](./or.md) | 逻辑或操作 |
|[not](./not.md) | 逻辑非操作 |
|[logical_and](./logical_and.md)| 用于对两个张量进行逐元素逻辑与运算 |
|[logical_or](./logical_or.md)| 用于对两个张量进行逐元素逻辑或运算 |
|[not](./not.md) | 将tensor的值按位取反。 |
|[invert](./invert.md) | 将tensor每个值按比特位进行翻转。 |
|[lshift](./lshift.md) | 根据给定值 将tensor张量进行左移位。 |
|[rshift](./rshift.md) | 根据给定值 将tensor张量进行右移位。 |
|[xor](./xor.md) | 计算两个元素的异或值。 |

```{toctree}
:maxdepth: 3
:hidden:
and.md
or.md
not.md
logical_and.md
logical_or.md
not.md
invert.md
lshift.md
rshift.md
xor.md
