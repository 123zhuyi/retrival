---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.4"
title: 原子操作粒度--连续段 E>=64 用向量 atomic_add，禁止标量循环 atomic
description: 连续输出段用 tl.atomic_add(ptr+arange,vals) 向量原子，标量循环 atomic 会让原子数爆炸
tags: [atomic_granularity, vector_atomic, scalar_atomic, simd]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-8 原子操作粒度

【优化维度】：算法｜跨 Block 归约
【泛化层级】：条件性（迁移条件：scatter 累加场景；失效边界：无累加的纯覆盖写）

<原则：向量 atomic 一次覆盖多个连续元素，原子数下降 1-2 个数量级；拆成标量循环 atomic 会破坏 SIMD 并爆炸原子数。与 [[atomic-add-cross-tile-reduction]] 配套：N4 是跨 tile 归并原语，本条是粒度选择。>

<骨架（占位名伪码）：
# 连续段 E>=64: 向量 atomic
offs_e = tl.arange(0, BLOCK_E)
mask = offs_e < E
tl.atomic_add(out_ptr + base + offs_e, vals, mask=mask)   # 一次覆盖多元素
# 禁止：标量循环 atomic
# for j in range(BLOCK_E):
#     tl.atomic_add(out_ptr + base + j, vals[j])  # 原子数爆炸
>

**坏实践（反例）**：标量循环 atomic 或 tl.get_element 拆标量
```python
for j in range(BLOCK_E):
    v = tl.get_element(vals, [j])        # 拆标量
    tl.atomic_add(out_ptr + base + j, v) # 标量 atomic
```
后果：原子数爆炸，性能骤降。✅ 正解即向量 atomic_add。

**已知实例**：index-computation G5（scatter 累加向量 atomic）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 相关主题: [Embedding Backward Scheduling 优化聚合](../compute-pipeline/traj-embedding-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](../compute-pipeline/work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关
- 相关主题: [向量化读写与 Coalesced Memory Access](../memory-sram/vectorized-coalesced-io.md) — 主题相关：共享 simd
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [Normalization Compute 优化聚合](traj-normalization-compute.md) — 主题相关：共享 simd

<!-- okf:related:end -->
