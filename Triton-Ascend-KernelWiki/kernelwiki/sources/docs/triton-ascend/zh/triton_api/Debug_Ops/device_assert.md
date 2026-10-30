---
id: doc-triton-ascend-zh-triton-api-debug-ops-device-assert
title: triton.language.deviceassert
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Debug_Ops/device_assert.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Debug_Ops/device_assert.md
content_sha256: 4df8ec526cf53dad59f104c24a03c15dd74f6fd0d27c1e31e6791456b205a2d5
snapshot_body_sha256: 4df8ec526cf53dad59f104c24a03c15dd74f6fd0d27c1e31e6791456b205a2d5
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- control
operators:
- device_assert
---

# triton.language.device_assert

**使用`device_assert`需要将环境变量`TRITON_DEBUG`的值设置为非`0`才能生效。**

## 1. 函数概述

`device_assert` 用于在GPU运行时从设备端进行断言检查，如果条件不满足则输出错误信息。

```python
triton.language.device_assert(cond, msg='', _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `cond` | `bool` | 必需 | 运行时需要断言的条件表达式 |
| `msg` | `str` | `''` | 断言失败时显示的错误消息 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

### 2.3 使用方法

```python
import triton.language as tl

@triton.jit
def basic_device_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 基本断言：检查程序ID
    pid = tl.program_id(0)
    tl.device_assert(pid >= 0, "Program ID must be non-negative")

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)

    # 检查数据有效性（比如检查张量中没有负值）
    tl.device_assert(tl.min(x) >= 0, "All values must be non-negative")
```
