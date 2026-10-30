---
id: doc-triton-ascend-zh-triton-api-extension-ops-parallel
title: triton.language.parallel
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Extension_Ops/parallel.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Extension_Ops/parallel.md
content_sha256: e91ca78017bb22d1bd667d0596b45d64f4b3e156cab70c2d4143305029ddb90a
snapshot_body_sha256: e91ca78017bb22d1bd667d0596b45d64f4b3e156cab70c2d4143305029ddb90a
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
operators:
- parallel
---

# triton.language.parallel

## 1. 函数概述

`parallel` 是一个专门用于多核心并行执行的迭代器，继承自 `range` 类，提供显式的多核心并行语义。

```python
triton.language.parallel(arg1, arg2=None, step=None, num_stages=None,
                         loop_unroll_factor=None, bind_sub_block: bool = False,
                         _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `arg1` | `int` /`constexpr`| 必需 | 起始值（单参数时作为结束值，从0开始） |
| `arg2` | `int`/`constexpr` | - | 结束值（不包含在范围内） |
| `step` | `int` /`constexpr`| `1` | 每次迭代的步长增量 |
| `num_stages` | `int` | - | 流水线阶段数（同时执行的迭代数量） |
| `loop_unroll_factor` | `int` | - | 循环展开因子（<2表示不展开） |
| `bind_sub_block` | `bool` | `False` | **关键参数**：绑定到子块，启用多核心并行执行 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

> **注意**：`parallel` 相比于 `range` 移除了以下参数：
>
> - `disallow_acc_multi_buffer`
> - `flatten`
> - `warp_specialize`
> - `disable_licm`

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend A2/A3 | ✓ | ✓ | ✓ | ×|×| × | × | ✓ | × | × | × | × | × |

### 2.3 特殊限制说明

`bind_sub_block` 为真时在IR中并体现出跟`range`的区别，功能是否实现待验证。

## 3. 使用方法

```python
@triton.jit
def parallel_kernel(
    input_ptr,
    output_ptr0,
    output_ptr1,
    pd_ptr,
    n_elements : tl.constexpr,
):
    x = tl.arange(0,n_elements)
    x0 = x // 4
    x1 = x % 4

    a_ptr = input_ptr + x0
    b_ptr = input_ptr + x0
    for i in tl.parallel(0, 3, 1, bind_sub_block = False):
        a_ptr += x0
        b_ptr += x0
    a_ptr += x1
    b_ptr += x1
    val = tl.load(a_ptr + 0)
    tl.store(output_ptr0 + x,val)
    val = tl.load(b_ptr)
    tl.store(output_ptr1 + x,val)
```
