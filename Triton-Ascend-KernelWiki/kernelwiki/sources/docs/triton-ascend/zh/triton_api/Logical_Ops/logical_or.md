---
id: doc-triton-ascend-zh-triton-api-logical-ops-logical-or
title: triton.language.tensor.logicalor
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Logical_Ops/logical_or.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Logical_Ops/logical_or.md
content_sha256: ba2cdd389c030ddc83b9f5298f6e63e7ea050c161c9bcb394f9a3be2797317d2
snapshot_body_sha256: ba2cdd389c030ddc83b9f5298f6e63e7ea050c161c9bcb394f9a3be2797317d2
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- math
operators:
- logical_or
hardware_features:
- fp8
---

# triton.language.tensor.logical_or

## 1. OP 概述

简介：用于对两个张量进行逐元素逻辑或运算

```python
x.logical_or(y)
```

作为`tensor`的成员函数调用, 如`x0.logical_or(x1)`。

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `input` | `tensor` | 张量数据, 左操作数, 代表要进行比较的主数据 |
| `other`   | `tensor` | 张量数据, 右操作数, 与`input`逐元素进行逻辑或 |
| `_builder` | - | 保留参数，暂不支持外部调用 |

返回值：
`tl.tensor`：同`input`的shape的张量

### 2.2 支持规格

#### 2.2.1 DataType 支持

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | × | × | × | × | × | × | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

结论：在 DataType 方面, Ascend相比GPU额外增加了对整型、浮点型（除fp64,fp8）的支持。

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| -------- | ---------------------- |
| GPU    | 无限制 |
| Ascend A2/A3 | 无限制 |

结论：在 Shape 方面, GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

无。

### 2.4 使用方法

以下示例实现了对三维张量`x0`、`x1`做逻辑或运算：

```python
@triton.jit
def triton_logical_or_3d(in_ptr0, in_ptr1, out_ptr0, XB, YB, ZB, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L) + tl.program_id(0) * XB
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * YB
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * ZB
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```
