---
id: doc-triton-ascend-zh-triton-api-reduction-ops-index
title: 归约操作
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Reduction_Ops/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Reduction_Ops/index.md
content_sha256: 1c45341704d2d63487a755af1c9afe2ef07db54887728f395f35a91809fb4752
snapshot_body_sha256: 1c45341704d2d63487a755af1c9afe2ef07db54887728f395f35a91809fb4752
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- math
---

# 归约操作

|api|简要说明|
|--|--|
|[argmax](./argmax.md) |返回沿指定 axis 轴上 input 张量中所有元素的最大索引 |
|[argmin](./argmin.md) |返回沿指定 axis 轴上 input 张量中所有元素的最小索引 |
|[max](./max.md) |返回沿指定 axis 轴上 input 张量中所有元素的最大值 |
|[min](./min.md) |返回沿指定 axis 轴上 input 张量中所有元素的最小值 |
|[reduce](./reduce.md) |将 combine_fn 应用于沿指定 axis 的 input 张量中的所有元素 |
|[sum](./sum.md) |返回 input 张量中，沿指定 axis 的所有元素的总和 |
|[xor_sum](./xor_sum.md) |返回 input 张量中，沿指定 axis 的所有元素的异或和 |

```{toctree}
:maxdepth: 3
:hidden:
argmax.md
argmin.md
max.md
min.md
reduce.md
sum.md
xor_sum.md
