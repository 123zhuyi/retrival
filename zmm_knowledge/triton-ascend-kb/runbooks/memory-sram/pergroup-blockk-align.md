---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)
description: pergroup 量化时 BLOCK_K 整除 group_size 保证每 K-tile 落单一 group，scale 只取一行
tags: [pergroup, blockk_align, group_size, single_row_scale]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-8 pergroup BLOCK_K 对齐 group_size

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：pergroup 量化；失效边界：pertensor 无 group 概念）

<原则：pergroup 时 BLOCK_K 整除 group_size（group_size 为 2 的幂），保证每个 K-tile 落在单一 group，scale 只取一行 scale[k//gs, :]。选 BLOCK_K=min(group_size,128)。与 [[pow2-block-constexpr]] 配套（group_size 本身是 2 的幂）。>

<骨架（占位名伪码）：
BLOCK_K = min(group_size, 128)   # 整除 group_size
# kernel 内 K-loop
for k_start in range(0, K, BLOCK_K):
    g_idx = k_start // group_size   # 单一 group
    scale_row = tl.load(scale_ptr + g_idx * N + offs_n)
    w_dq = (w_block.to(f32) + off_block.to(f32)) * scale_row
>

**坏实践（反例）**：BLOCK_K 不整除 group_size
```python
BLOCK_K = 128  # group_size=64 时，一个 K-tile 跨两个 group
# 需要在 tile 内按 group 拆分，scale 取多行，逻辑复杂且慢
```
后果：K-tile 跨 group，scale 索引复杂化，性能下降。✅ 正解即 BLOCK_K 整除 group_size。

**已知实例**：weight-quant-matmul W4

<!-- okf:related:start -->

# 相关

- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 group_size
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 同属优化知识，主题相关
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](../portability/exp_rope_llama4_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](bf16-fused-dequant-blockk128.md) — 同属优化知识，主题相关
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 同属优化知识，主题相关
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](row-width-512b-align.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
