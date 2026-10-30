---
id: doc-triton-ascend-zh-triton-api-compiler-hint-ops-assume
title: triton.language.assume
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Compiler_Hint_Ops/assume.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Compiler_Hint_Ops/assume.md
content_sha256: dacf5c6f2edd42c942823d1060d3359451c82349a739e38aa76ee1a032346ea9
snapshot_body_sha256: dacf5c6f2edd42c942823d1060d3359451c82349a739e38aa76ee1a032346ea9
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- control
operators:
- assume
---

# triton.language.assume

## 1. 函数概述

`assume` 用于向编译器提供条件假设信息，允许编译器基于已知为真的条件进行优化。这是一个编译器提示操作，不会在运行时检查条件。

```python
triton.language.assume(cond, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `cond` | `bool` | 必需 | 编译器可以假设为真的条件表达式 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

### 2.3 使用方法

`assume` 操作允许开发者在确保正确性的前提下，帮助编译器生成更高效的代码。

```python
import triton.language as tl

@triton.jit
def basic_assume_example(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # 假设BLOCK_SIZE是2的幂次，编译器可以基于此优化除法运算
    tl.assume((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)

    # 编译器知道BLOCK_SIZE是2的幂次，可以优化除法为移位操作
    result = x // BLOCK_SIZE + y % BLOCK_SIZE
    tl.store(y_ptr + offsets, result)
```
