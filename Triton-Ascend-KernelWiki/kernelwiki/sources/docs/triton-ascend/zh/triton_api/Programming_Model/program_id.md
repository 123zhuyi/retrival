---
id: doc-triton-ascend-zh-triton-api-programming-model-program-id
title: triton.language.programid
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Programming_Model/program_id.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Programming_Model/program_id.md
content_sha256: a765ed76b8af7db1bee6d5df9b442350be2513e4d20bf4b0c744d261a8d25b5a
snapshot_body_sha256: a765ed76b8af7db1bee6d5df9b442350be2513e4d20bf4b0c744d261a8d25b5a
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- control
operators:
- program_id
---

# triton.language.program_id

## 1. OP 概述

简介：返回当前程序实例沿给定 axis 的 ID。
函数原型：

```python
triton.language.program_id(axis)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `axis` | `int` |  3D 启动网格的轴。必须是 0、1 或 2 |

返回值：
由轴的值组成的tl.tensor

### 2.2 支持规格

#### 2.2.1 DataType 支持

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | √ | × | × | × | × | × | × | × | × | × | × |
| Ascend A2/A3 | × | × | √ | × | × | × | × | × | × | × | × | × | × |

#### 2.2.2 Shape 支持

无相关设置

### 2.3 特殊限制说明

无

### 2.4 使用方法

在triton kernel中会用到，用于获取PID

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
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
    Y = tl.load(y_ptr + idx)

    ...

    tl.store(output_ptr + idx, ret)
```
