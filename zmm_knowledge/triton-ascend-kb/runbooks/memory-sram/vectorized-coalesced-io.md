---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 向量化读写与 Coalesced Memory Access
description: trailing/E/D 连续维度必须用 tl.arange+mask 向量 load/store，索引张量也按 BLOCK_M 向量加载
tags: [vectorized_io, coalesced, arange_load, simd]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-9 向量化读写 coalesced

【优化维度】：内存｜访存效率
【泛化层级】：Triton通用（迁移条件：所有 load/store；无失效边界）

<原则：标量加载索引或标量 atomic 会破坏 SIMD，原子数与元素数相同，加速比从 0.3-0.6x 跌至 0.05-0.15x。连续维度必须 tl.arange+mask 向量 load/store，索引张量也按 BLOCK_M 向量加载。与 [[input-contiguous]] 配套。>

<骨架（占位名伪码）：
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < total
vals = tl.load(src_ptr + offs, mask=mask)        # 向量 load
# 索引张量也向量加载
idx = tl.load(index_ptr + offs_m, mask=mask_m).to(tl.int32)
tl.store(dst_ptr + offs, vals, mask=mask)         # 向量 store
>

**坏实践（反例）**：标量加载/标量 atomic
```python
for i in range(BLOCK):
    v = tl.load(ptr + i)           # 标量 load，破坏 SIMD
    tl.atomic_add(out + idx[i], v) # 标量 atomic，原子数爆炸
```
后果：加速比 0.05-0.15x。✅ 正解即向量化 load/store。

**已知实例**：index-computation G2

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_007` — 性能避坑 （收益 -40%）
  - 负性能提升表示stride访问相比最优方案的性能损耗；910b2/910b3平台对stride访问的硬件惩罚显著，2D张量（1024x1024）的stride访问耗时是连续一维访问的1.4倍以上

<!-- okf:related:start -->

# 相关

- 相关主题: [原子操作粒度--连续段 E>=64 用向量 atomic_add，禁止标量循环 atomic](../algorithm-numerics/atomic-granularity.md) — 主题相关：共享 simd
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 主题相关：共享 coalesced

<!-- okf:related:end -->
