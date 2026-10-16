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

- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
