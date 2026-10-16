---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: per-channel 参数标量 load 路径，避免向量化 int 除法降级
description: BLOCK 整体落在同一 channel 时用标量索引加载 weight/bias，禁止对 tl.arange 向量做 // 或 % 求 channel
tags: [scalar_load, vector_division, channel_index, weight_bias]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-1 per-channel 参数标量 load 路径

【优化维度】：计算｜Scalar 削减
【泛化层级】：Triton通用（迁移条件：BLOCK_SIZE ≤ channelLen，即整个 block 落在同一 channel；失效边界：BLOCK_SIZE > channelLen 时需在 host 侧缩 BLOCK 以保证标量路径）

<原则：Ascend NPU 上 int32 向量除法退化为标量循环，BLOCK=1024 时每个 block 执行 1024 次标量除法。改用标量索引（对单标量做一次除法）可避免向量降级；且 BLOCK_SIZE ≥ channelLen 时标量 load 仍正确工作（每次循环加载同一 channel）。关联硬件依据：Vector 单元对 int32 除法无原生向量指令，退化为 Scalar 循环。>

<骨架（占位名伪码，可代入）：
channelIdx = tileOffset // channelLen        # 标量除法（tileOffset 为标量，无向量降级）
w = tl.load(weight_ptr + channelBase + channelIdx)
b = tl.load(bias_ptr + channelBase + channelIdx)
x_norm = x_norm * w + b
>

**坏实践（反例）**：对 `tl.arange` 产生的向量做 `//` 求 channel 后向量化 load
```python
channel_vec = offsets // channelLen          # 向量化 int 除法，Ascend 上严重 scalar 化
w_vec = tl.load(weight_ptr + channelBase + channel_vec, mask=mask)
```
后果：实测加速比 0.098x（vs 标量 load 1.198x，性能差约 12 倍）。✅ 正解即本条标量 load 路径。

**已知实例**：GroupNormSwish §2.3 L3.1（`triton-op-generator/template/normalization.md` N3）

<!-- okf:related:start -->

# 相关

- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 scalar_load
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](int-compare-cast-float32.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
