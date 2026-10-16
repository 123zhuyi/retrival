---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 按输出元素数（而非输入）分配 grid，且钳制到 num_cores
description: 数学计算类算子输入输出规模差异大时，必须按输出元素总数分配 grid，否则空跑或负载不均
tags: [grid_by_output, output_count, math_compute, no_oversubscribe]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-7 按输出元素数分配 grid

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：输入输出规模差异大的算子（reduce/FFT）；失效边界：输入输出等规模时按输入分配即可）

<原则：数学计算算子输入输出规模往往差异大（reduce 输入>>输出，FFT 输出维度≠输入），按输入分配会导致空跑或负载不均。必须按输出元素总数分配，再钳制到核数。与 [[grid-cap-to-core-num]] 配套。>

<骨架（占位名伪码）：
total_out = <输出元素总数>   # 非输入元素数
grid_size = min(total_out, num_cores)
grid = (grid_size,)
>

**坏实践（反例）**：按输入元素数分配 grid
```python
grid = (min(total_in, num_cores),)   # reduce 输入>>输出，多余 core 空跑
```
后果：reduce 类算子输入远大于输出，按输入分配导致部分 core 处理完后空等。✅ 正解即按输出元素数分配。

**已知实例**：math-compute M5（Sum 模式 A grid=total_out、Sort grid=cdiv(n_rows,BLOCK_ROWS)、FFT grid=min(total_seq×tiles,num_cores)）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 实践案例: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](../../ops/moe-finalize-routing-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 同属优化知识，主题相关
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 主题相关：共享 no_oversubscribe
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
