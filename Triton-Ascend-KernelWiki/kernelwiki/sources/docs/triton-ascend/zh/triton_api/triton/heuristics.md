---
id: doc-triton-ascend-zh-triton-api-triton-heuristics
title: triton.heuristics
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/triton/heuristics.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/triton/heuristics.md
content_sha256: 4f00568fdd5c1687f4389bc7d724042b7017a74fb768ce2b6dd73150cbe1ca10
snapshot_body_sha256: 4f00568fdd5c1687f4389bc7d724042b7017a74fb768ce2b6dd73150cbe1ca10
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
operators:
- heuristics
---

# triton.heuristics

```python
triton.heuristics(values)
```

用于指定如何计算某些元参数值的装饰器。这在自动调优成本过高或不适用的情况下非常有用。

```python
@triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size  最小的 2 的幂 >= x_size
```

**参数：** `values (dict[str, Callable[[list[Any]], Any]]**)` - 包含元参数名称和计算元参数值的函数的字典。每个这样的函数都接受一个位置参数列表作为输入。
