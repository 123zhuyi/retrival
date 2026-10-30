---
id: doc-triton-ascend-zh-triton-api-triton-config
title: triton.Config
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/triton/Config.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/triton/Config.md
content_sha256: 1290ab522ff8c6a6d124c585dcbdc4865b95cfb35621a20fc027093420ac07f0
snapshot_body_sha256: 1290ab522ff8c6a6d124c585dcbdc4865b95cfb35621a20fc027093420ac07f0
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
operators:
- Config
---

# triton.Config

```python
class triton.Config(self, kwargs, num_warps, num_stages, num_ctas, maxnreg, pre_hook)
```

表示自动调优可能尝试的内核配置的对象

**变量：**

- kwargs – 1 个元参数字典，用于作为关键字参数传递给内核。

- num_warps – 在为 GPU 编译时内核使用的线程数。例如，如果 num_warps=8，则每个内核实例将自动并行化，使用 8 * 32 = 256 个线程协作执行。

- num_stages – 编译器在软件流水线循环时应使用的阶段数。对于 SM80+ GPU 上的矩阵乘法工作负载非常有用。

- num_ctas - 块集群中的块数。仅适用于 SM90+。

- maxnreg - 单个线程可以使用的最大寄存器数。对应于 ptx 的 .maxnreg 指令。并非所有平台都支持。

- pre_hook – 在调用内核之前将被调用的函数。该函数的参数是 args。

```python
__init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None)
```

**方法：**

| init(self, kwargs[, num_warps, ...]) |
|-----|
|all_kwargs (self)|
