---
id: doc-triton-ascend-zh-triton-api-shape-manipulation-ops-join
title: triton.language.join
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Shape_Manipulation_Ops/join.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Shape_Manipulation_Ops/join.md
content_sha256: 0778b122e70b44083865872a4dbfb4a6fe74f02b098985bb08a38aa7468efc74
snapshot_body_sha256: 0778b122e70b44083865872a4dbfb4a6fe74f02b098985bb08a38aa7468efc74
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- conversion
operators:
- join
---

# triton.language.join

## 1 功能作用说明

将两个相同形状的输入张量沿着新的最小维度连接，输出张量比输入张量多一个维度，大小为2，保持其他维度不变。

**语法：**

- `triton.language.join(x, y)` - 函数调用形式
- `x.join(y)` - 成员函数形式

**功能：**

- 将两个相同形状的输入张量沿着新的最小维度连接
- 输出张量比输入张量多一个维度，大小为2
- 保持其他维度不变

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| x | tensor | 是 | 第一个输入张量 |
| y | tensor | 是 | 第二个输入张量 |

**返回值：**

- **类型：** tensor
- **形状：** 输入tensor广播后的形状加上一个大小为2的维度
- **数据类型：** 与输入张量相同
- **内存布局：** 在新增维度上堆叠x和y

**约束条件：**

- 两个输入张量必须具有可以广播到相同形状的形状和数据类型

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape支持表

支持任意维度数、任意形状大小。

### 2.4 特殊限制说明

无

### 2.5 使用方法

```python
import torch
import triton
import triton.language as tl

@triton.jit
def join_example(out_ptr):
    # 创建两个2x3的张量
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.full([2, 3], 1.0, dtype=tl.float32)

    # 连接，变成2x2x3
    z = tl.join(x, y)

    # 将结果写回外部张量
    offs = (
        tl.arange(0, 2)[:, None, None] * (2 * 3)
        + tl.arange(0, 2)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, z)

## 调用示例
out = torch.empty((2, 2, 3), dtype=torch.float32, device="npu")
join_example[(1,)](out)
print(out.shape)  # 输出: torch.Size([2, 2, 3])
```
