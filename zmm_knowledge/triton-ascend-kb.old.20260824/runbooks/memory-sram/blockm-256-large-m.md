---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: M>=256 时 BLOCK_M=256（总 block-iter 减半）
description: 大 M 场景 BLOCK_M=256 让 L0A/L0C 满载，总 block-iter 减半，改善 cube 利用率
tags: [blockm_256, large_m, l0a_l0c, block_iter_halve]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-10 BLOCK_M=256 对大 M

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：M>=256 的大 batch matmul；失效边界：M<256 时用较小 BLOCK_M）

<原则：M>=256 时 BLOCK_M=256（L0A/L0C 满载可编译），总 block-iter 减半，更大 dot 改善 cube 利用率。与 [[swizzle2d-block-reorder]] 配套提升 L2 复用。>

<骨架（占位名伪码）：
if M >= 256:
    BLOCK_M = 256   # L0A/L0C 满载，block-iter 减半
else:
    BLOCK_M = 128   # 小 M 用较小 block
>

**坏实践（反例）**：大 M 用小 BLOCK_M
```python
BLOCK_M = 128   # M=1024 时需 8 个 block-iter，cube 利用率低
```
后果：大 M case 性能差（0.27x -> 0.5x+ 提升空间未用）。✅ 正解即 M>=256 用 BLOCK_M=256。

**已知实例**：weight-quant-matmul W8（大 M case 0.27->0.5+，bf16/medium 0.48->0.65+）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [swizzle2d 块重排提升 L2 复用](../compute-pipeline/swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](ub-aware-block-adaptive.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
