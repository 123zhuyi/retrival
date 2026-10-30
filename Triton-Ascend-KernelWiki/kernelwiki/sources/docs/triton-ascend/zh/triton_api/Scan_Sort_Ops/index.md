---
id: doc-triton-ascend-zh-triton-api-scan-sort-ops-index
title: 扫描/排序操作
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Scan_Sort_Ops/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Scan_Sort_Ops/index.md
content_sha256: 837dff17181b4da6de603ef75f38d6390ab84213e9213e5214f56a95b8eb0add
snapshot_body_sha256: 837dff17181b4da6de603ef75f38d6390ab84213e9213e5214f56a95b8eb0add
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- index
---

# 扫描/排序操作

| api | 简要说明 |
| --- | --- |
| [associative_scan](./associative_scan.md) | 沿指定 axis 将 combine_fn 应用于 input 张量的每个元素和携带的值，并更新携带的值 |
| [cumprod](./cumprod.md) | 返回沿指定 axis 的 input 张量中所有元素的累积乘积 |
| [cumsum](./cumsum.md) | 返回沿指定 axis 的 input 张量中所有元素的累积和 |
| [histogram](./histogram.md) | 基于 input 张量计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0 |
| [sort](./sort.md) | 沿着指定维度对张量进行排序 |
| [topk](./topk.md) | 返回沿指定维度的前 k 个最大元素 |

```{toctree}
:maxdepth: 3
:hidden:
associative_scan.md
cumprod.md
cumsum.md
histogram.md
sort.md
topk.md
```
