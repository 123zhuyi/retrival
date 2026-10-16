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

- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 cache_hit
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 l2_reuse

<!-- okf:related:end -->
