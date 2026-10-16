---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.3"
title: precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）
description: pergroup gs<=64 或大 M 中等 N 时改用 precompute 路径（先反量化再纯 matmul），其余用 fused
tags: [precompute, dispatch, dequant_split, fused_vs_precompute]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-2 precompute 分派决策

【优化维度】：元技能｜设计决策
【泛化层级】：条件性（迁移条件：pergroup gs<=64 或 (M>=1024 且 N<=8192)；失效边界：N>8192 大 case precompute 内存主导反而回退）

<原则：pergroup gs<=64 时 fused 被迫 BK=32/64，或大 M 中等 N 时，改用 precompute 路径：单独 _dequant_kernel 把 int8 权重反量化成 low_dtype [K,N]，再 _matmul_kernel 纯 matmul（BK=128 不受 gs 约束 + 移除 K-loop 反量化减 sync）。其余用 fused。是 [[multi-strategy-dispatch]] 的量化 matmul 专用版。>

<骨架（占位名伪码）：
if group_size <= 64 or (M >= 1024 and N <= 8192):
    # precompute 路径
    _dequant_kernel[grid1](w_int8, w_dq, ...)     # 先反量化成 [K,N]
    _matmul_kernel[grid2](x, w_dq, out, BLOCK_K=128, ...)  # 纯 matmul
else:
    _fused_kernel[grid](x, w_int8, scale, out, ...)  # fused 反量化+matmul
# 禁止：N>8192 用 precompute（w_dq 内存 128MB@N=16384 主导）
>

**坏实践（反例）**：N>8192 用 precompute 或 precompute dequant 用 per-element gather
```python
if N > 8192:
    _dequant_kernel[grid](...)  # w_dq [K,N] 内存 128MB 主导，反而回退
# dequant 内 offs_k//gs [BK,1] gather -> MLIR assert crash
```
后果：N>8192 回退（opt_iter_9 教训）；gather 触发编译器 crash。✅ 正解即按 gs/M/N 分派 + single-row scale。

**已知实例**：weight-quant-matmul W10（gs<=64 case 0.27/0.48->0.56/0.73，geomean 0.6814->0.7571）

<!-- okf:related:start -->

# 相关

- 相关主题: [Moe Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md) — 同属优化知识，主题相关
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](multi-strategy-dispatch.md) — 主题相关：共享 dispatch

<!-- okf:related:end -->
