---
id: doc-triton-ascend-zh-triton-api-compiler-hint-ops-multiple-of
title: triton.language.multipleof
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Compiler_Hint_Ops/multiple_of.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Compiler_Hint_Ops/multiple_of.md
content_sha256: 08f9f474db1f22aed867230ee194d5cd81df2e844c63a30083a1f59fa67d9163
snapshot_body_sha256: 08f9f474db1f22aed867230ee194d5cd81df2e844c63a30083a1f59fa67d9163
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- control
operators:
- multiple_of
---

# triton.language.multiple_of

## 1. 函数概述

`multiple_of` 用于向编译器声明输入张量中的第一个值是某个数的倍数。

```python
triton.language.multiple_of(input, values, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `input` | `Tensor` | 必需 | 输入张量，其值都是某个数的倍数 |
| `values` | `constexpr[int]` 或 `list[constexpr[int]]` | 必需 | 声明输入值是这些数的倍数（单个整数或按维度的整数序列） |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

**`values`描述着每个维度第一个值的可除性特征，所以`values` 的维度要与`input` 的维度相同。
注意当`shape`的最后一维为`1`时出现的降维情况。**

如：二维 `input`对应通用`values`入参为`[1,1]`。

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

### 2.4 使用方法

```python
@triton.jit
def basic_multiple_of_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # 声明输入张量的第一个值是BLOCK_SIZE的倍数
    input_data = tl.multiple_of(input_data, [BLOCK_SIZE])
```
