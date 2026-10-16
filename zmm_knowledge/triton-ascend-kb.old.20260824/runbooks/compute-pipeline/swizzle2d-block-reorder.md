---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: swizzle2d 块重排提升 L2 复用
description: tl.swizzle2d 重排块访问顺序让相邻 program 共享输入块，提升 L2 cache 复用
tags: [swizzle2d, block_reorder, l2_reuse, cache_hit]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-10 swizzle2d 块重排

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：2D block 划分的 matmul，多 program 共享输入块；失效边界：1D 划分或无共享）

<原则：tl.swizzle2d 重排块访问顺序，让相邻 program 处理相邻 block，共享同一输入块（如同一行 K），提升 L2 cache 复用。配合 BLOCK_K=128 减半 K 迭代。>

<骨架（占位名伪码）：
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
pid_m, pid_n = tl.swizzle2d(pid, num_pid_m, num_pid_n, GROUP_SW)
# 相邻 pid 处理相邻 (m,n) block，共享 K 行输入 -> L2 复用
>

**坏实践（反例）**：线性 block 顺序
```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n   # 线性顺序，相邻 program 不共享输入
```
后果：L2 cache 复用差，HBM 读取多。✅ 正解即 swizzle2d 重排。

**已知实例**：weight-quant-matmul W9（配合 BLOCK_K=128）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](../memory-sram/blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](../memory-sram/ub-aware-block-adaptive.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 l2_reuse
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 l2_reuse
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](traj-general_perf-block_size.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
