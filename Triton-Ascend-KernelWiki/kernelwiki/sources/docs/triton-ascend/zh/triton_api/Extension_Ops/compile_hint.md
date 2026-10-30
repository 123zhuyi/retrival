---
id: doc-triton-ascend-zh-triton-api-extension-ops-compile-hint
title: triton.language.compilehint
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Extension_Ops/compile_hint.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Extension_Ops/compile_hint.md
content_sha256: dbe5ccbdca18f826abeb4e4675abf65b660294fa3888d73888139bba37435623
snapshot_body_sha256: dbe5ccbdca18f826abeb4e4675abf65b660294fa3888d73888139bba37435623
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
operators:
- compile_hint
---

# triton.language.compile_hint

## 1.函数概述

`compile_hint` 是一个编译器提示（hint）机制，允许用户为张量附加元数据信息，这些信息会被传递到编译器后端，用于指导优化和代码生成。

```python
triton.language.compile_hint(ptr, hint_name, hint_val=None, _builder=None)
```

## 2.规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|------|------|
| `ptr` | `tensor` | 必需 | 需要附加提示的张量对象 |
| `hint_name` | `str` `constexpr` | 必需 | 提示的名称标识符（必须为字符串） |
| `hint_val` | `None` `bool` `int` `constexpr` `list` | `None` | 提示的值，支持多种类型 |
| `_builder` | - | `None` | 保留参数，暂不支持外部调用 |

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.3 特殊限制说明

1. **hint_name 必须为字符串类型**：不能传入其他类型作为提示名称
2. **list 参数仅支持整数数组**：元素须为整数（`int` 或 `constexpr` 整数），不支持浮点数或混合类型的列表
3. **非侵入式设计**：`compile_hint` 不改变计算语义，仅添加元数据
4. **同一张量可多次标注**：同一个张量可以附加多个不同名称的提示

### 2.4 使用方法

```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.compile_hint(tmp0, "hint_a")
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```
