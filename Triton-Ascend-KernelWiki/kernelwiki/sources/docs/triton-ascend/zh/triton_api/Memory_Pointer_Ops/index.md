---
id: doc-triton-ascend-zh-triton-api-memory-pointer-ops-index
title: 内存/指针操作
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Memory_Pointer_Ops/index.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Memory_Pointer_Ops/index.md
content_sha256: 26cac50df79e93fd35d55bf165654a2cb6f8ce4b2cde214ce6717ee61202bc8c
snapshot_body_sha256: 26cac50df79e93fd35d55bf165654a2cb6f8ce4b2cde214ce6717ee61202bc8c
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- conversion
---

# 内存/指针操作

|api|简要说明|
|--|--|
|[load](./tl.load.md) |返回一个张量，其值从由指针定义的内存位置加载|
|[store](./tl.store.md) |将数据张量存储到由指针定义的内存位置|
|[make_block_ptr](./tl.make_block_ptr.md) |返回指向父张量中某个块的指针|
|[advance](./tl.advance.md) |推进一个块指针|
|[load_tensor_descriptor](./load_tensor_descriptor.md) | 从张量描述符加载数据块 |
|[make_tensor_descriptor](./make_tensor_descriptor.md) | 创建张量描述符对象 |

```{toctree}
:maxdepth: 3
:hidden:
tl.load.md
tl.store.md
tl.make_block_ptr.md
tl.advance.md
load_tensor_descriptor.md
make_tensor_descriptor.md
