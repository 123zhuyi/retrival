---
id: doc-triton-ascend-zh-triton-api-random-number-generation-randint4x
title: triton.language.randint4x
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/triton_api/Random_Number_Generation/randint4x.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- triton-api
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/triton_api/Random_Number_Generation/randint4x.md
content_sha256: 4115587dd79fd5805b2bf1d1d2ce48b87eac7e1f159864e9c453d8d384371220
snapshot_body_sha256: 4115587dd79fd5805b2bf1d1d2ce48b87eac7e1f159864e9c453d8d384371220
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
kernel_types:
- random
operators:
- randint4x
---

# triton.language.randint4x

## 1. OP 概述

简介：给定 1 个 seed 标量和 1 个 offset 块，返回 4 个 int32 类型的随机块。
Triton 的 Philox 伪随机数生成器的最高效入口点。
原型：

```python
triton.language.randint4x(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `seed`        |`int`或 `tensor`          | 用于生成随机数的种子                                                   |
| `offset`       | `int`或 `tensor`      | 用于生成随机数的偏移量                     |
| `n_rounds`     | `constexpr`，默认值为10   | Philox 算法的迭代轮数 |

返回值：
4 个 int32 类型的随机块，每个块的shape与offset相同

### 2.2 支持规格

#### 2.2.1 DataType 支持

输入seed的type：

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |

#### 2.2.2 Shape 支持

无特殊要求

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

### 2.4 使用方法

以下示例实现了offset标量时对randint4x的调用：

```python
@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask) # 存储随机数

y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()
kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)
```

以下例子实现了offset非标量时对randint4x的调用，其中存储所用的tensor大小是offset的4倍：

```python
@triton.jit
def triton_randint4x1d(out_ptr, seed, L: tl.constexpr):
 idx = tl.arange(0, L)
 rnd0, rnd1, rnd2, rnd3 = tl.randint4x(seed, idx)
 pointers0 = out_ptr + idx
 pointers1 = out_ptr + L + idx
 pointers2 = out_ptr + 2 * L + idx
 pointers3 = out_ptr + 3 * L + idx
 tl.store(pointers0, rnd0)
 tl.store(pointers1, rnd1)
 tl.store(pointers2, rnd2)
 tl.store(pointers3, rnd3)
```
