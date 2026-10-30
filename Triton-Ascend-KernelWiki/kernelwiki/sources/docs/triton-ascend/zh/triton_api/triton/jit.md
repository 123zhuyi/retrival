---
id: doc-triton-ascend-zh-triton-api-triton-jit
title: triton.jit
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/triton/jit.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/triton/jit.md
content_sha256: 8c839aefd5c1dff9fee62f73cfa94315b223af0cb0f2b6edf5ac7c0007fa20e2
snapshot_body_sha256: 8c839aefd5c1dff9fee62f73cfa94315b223af0cb0f2b6edf5ac7c0007fa20e2
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
operators:
- jit
---

# triton.jit

```python
triton.jit(fn: T) -> JITFunction[T]
triton.jit(*, version=None, repr: Callable | None = None, launch_metadata: Callable | None = None, do_not_specialize: Iterable[int] | None = None, debug: bool | None = None, noinline: bool | None = None) -> Callable[[T], JITFunction[T]]
```

使用 Triton 编译器的 JIT 编译函数的装饰器。

- 注：当调用 JIT 编译的函数时，如果参数具有 `.data_ptr()` 方法和 `.dtype` 属性，则会隐式转换为指针。

- 注：此函数将在 GPU 上编译和运行。它只能访问以下内容：
    - Python 原语，
    - Triton 包内的内置函数，
    - 此函数的参数，
    - 其他 JIT 编译的函数。

**参数：**`fn (Callable)` - 要进行 JIT 编译的函数
