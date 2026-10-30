---
id: doc-triton-ascend-zh-triton-api-math-ops-mul
title: mul
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Math_Ops/mul.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Math_Ops/mul.md
content_sha256: 6ee874a009e6ed770ff4cd37e13fd957fad8d1e8f334e8949dcee0420a3c1e75
snapshot_body_sha256: 6ee874a009e6ed770ff4cd37e13fd957fad8d1e8f334e8949dcee0420a3c1e75
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- math
operators:
- mul
---

# mul

## 1. OP 概述

简介：四则运算 ‘*’,  无tl.mul调用方法

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor or Number`     |     第一个入参     |
| `y`       | `tensor or Number`     |     第二个入参     |

返回值：
`tl.tensor`：乘法结果

### 2.2 支持规格

#### 2.2.1 DataType 支持

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
|GPU| √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
|Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 无限制 |
| Ascend A2/A3 |无限制  |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 使用方法

以下示例实现了对输入张量 `in_ptr0, in_ptr1` 做乘法计算：

```python
@triton.jit
def triton_mul(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 * tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)
```

## 3.  特殊说明

Ascend A3 对比 GPU 缺失fp64的支持
