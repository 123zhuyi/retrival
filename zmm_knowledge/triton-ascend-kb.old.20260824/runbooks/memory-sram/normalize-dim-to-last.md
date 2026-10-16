---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 目标维度必须归一化到末维（permute->contiguous->处理->permute 回）
description: reduce/sort dim 非末维时，host 侧 permute 到末维再进 kernel，禁止 kernel 内处理多维非末维语义
tags: [dim_to_last, permute, reduce_dim, sort_dim, host_preprocess]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-3 目标维度归一化到末维

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：reduce/sort 目标维度非末维；失效边界：目标维度已是末维无须 permute）

<原则：kernel 内降为 2D (rows, target_dim) 逻辑能最大化向量化与并行度；多维 grid 在 reduce/sort 类算子上无收益且坐标解码 overhead 大。目标维度非末维时，host 侧 permute->contiguous->处理->permute 回。是 [[input-contiguous]] 的强化：G7 要求整体 contiguous，本条要求目标维归一化到末维。>

<骨架（占位名伪码）：
# host 侧 forward
if dim != x.ndim - 1:
    x = x.transpose(dim, -1)   # 目标维换到末维
    x = x.contiguous()         # 连续化
# kernel 处理 2D (rows, target_dim)
kernel[grid](x, out, ...)
if dim != x.ndim - 1:
    out = out.transpose(-1, dim)  # 换回
>

**坏实践（反例）**：kernel 内处理多维非末维
```python
# kernel 内用多维 grid + 坐标解码处理 dim != -1
# 坐标解码 overhead 大，向量化差
```
后果：多维 grid 在 reduce/sort 类算子上无收益，坐标解码 overhead 大。✅ 正解即 host 侧 permute 到末维。

**已知实例**：math-compute M1（Sort dim != ndim-1 路径 L1.2、Sum dim 归一化）

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](../../ops/moe-token-permute-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](../../ops/moe-token-permute-with-ep-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 同属优化知识，主题相关
- 相关主题: [数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）](../algorithm-numerics/math-normalize-precondition.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）](../portability/no-pytorch-in-forward.md) — 主题相关：共享 host_preprocess
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](host-flatten-to-2d.md) — 主题相关：共享 host_preprocess
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 host_preprocess
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
