---
id: doc-triton-ascend-zh-triton-api-math-ops-ceil
title: triton.language.ceil
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Math_Ops/ceil.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Math_Ops/ceil.md
content_sha256: 358d904e80d7dd349d52948ca91eca43f090932cae0d4e1767dfc647dc8c4f2a
snapshot_body_sha256: 358d904e80d7dd349d52948ca91eca43f090932cae0d4e1767dfc647dc8c4f2a
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- math
operators:
- ceil
---

# triton.language.ceil

## 1. OP 概述

简介：计算张量中每个元素的向上取整值

```python
triton.language.ceil(x, _semantic=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `x` | `tensor` | 张量数据 |
| `_semantic`   | - | 保留参数，暂不支持外部调用 |

返回值：
`out`：同`x`的shape的张量

### 2.2 支持规格

#### 2.2.1 DataType 支持

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | × | × | × | × | × | × | × | √ | √ | × | × |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

#### 2.2.2 Shape 支持

|        | 支持维度范围         |
| :---: | :---: |
| GPU    | 无限制 |
| Ascend | 无限制 |

结论：在 Shape 方面, GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失fp64的支持能力, 但多了fp16、bf16的支持能力, 支持整型输入。

### 2.4 使用方法

以下示例实现了对输入张量 `x` 做向上取整运算：

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.ceil(X)

    tl.store(output_ptr + idx, ret)
```
