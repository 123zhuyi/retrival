---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: kernel 内索引必须转 int32，避免 int64 降级
description: tl.program_id 和 tl.arange 结果必须 .to(tl.int32)，int64 索引触发地址计算降级
tags: [int32_index, int64_demote, scalar_demote, address_compute]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-3 kernel 内索引转 int32

【优化维度】：计算｜Scalar 削减
【泛化层级】：Triton通用（迁移条件：所有 kernel 内索引计算；无失效边界）

<原则：int64 标量会触发地址计算降级为标量循环；NPU 上 int32 索引更高效。tl.program_id/tl.arange 默认可能是 int64，必须显式转 int32。与 [[scalar-load-channel-index]] 同属 Scalar 削减。>

<骨架（占位名伪码）：
pid = tl.program_id(0).to(tl.int32)
block_start = pid * BLOCK_SIZE
offsets = (block_start + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
mask = offsets < total_elems
>

**坏实践（反例）**：保留 int64 索引
```python
pid = tl.program_id(0)              # int64
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # int64 地址计算降级
```
后果：地址计算退化为标量循环，性能下降。✅ 正解即 .to(tl.int32)。

**已知实例**：tensor-transform G5（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 同属优化知识，主题相关
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](int-compare-cast-float32.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
