---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应
description: 不同算法 live_buffers 差异大，必须按 UB 估算为每类 kernel 提供 BLOCK 选择函数，禁止共用单一函数
tags: [ub_budget, block_rows_adaptive, live_buffers, tile_size]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-5 UB-aware BLOCK 自适应

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：不同算子 live_buffers_count 差异大；失效边界：所有算子 live_buffers 相同时可共用）

<原则：UB 估算规则 BLOCK_ROWS * L * dtype_itemsize * live_buffers_count <= UB_TOTAL。不同算法 live_buffers_count 差异大（ext.sort ~1 buffer，软件 bitonic ~8 buffers，FFT butterfly 2-3 buffers），必须分别提供 UB-aware 的 BLOCK 选择函数。目标 grid=cdiv(n_rows,BLOCK_ROWS) <= num_cores。>

<骨架（占位名伪码）：
UB_TOTAL = 192 * 1024  # 192KB（按芯片/场景调整）
live_buffers = {"ext_sort": 1, "bitonic": 8, "fft": 3}[algo]
dtype_bytes = 4 if dtype == fp32 else 2
max_block_rows = UB_TOTAL // (L * dtype_bytes * live_buffers)
BLOCK_ROWS = min(max_block_rows, n_rows) if n_rows > num_cores else 1
>

**坏实践（反例）**：所有算子共用单一 BLOCK 函数
```python
def block_rows(L, dtype):       # 未考虑 live_buffers 差异
    return UB_TOTAL // (L * dtype_bytes)
# bitonic 8 buffers 用这个会 UB overflow
```
后果：live_buffers 多的算子 UB overflow，寄存器压力，性能退化。✅ 正解即按算法特征分别自适应。

**已知实例**：math-compute M4（Sort _block_rows_for_extsort UB=8192 fp32 / _block_rows_for_L 50KB 8 buffers 分离、FFT BLOCK_SIZE_B=64）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [swizzle2d 块重排提升 L2 复用](../compute-pipeline/swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 ub_budget
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
