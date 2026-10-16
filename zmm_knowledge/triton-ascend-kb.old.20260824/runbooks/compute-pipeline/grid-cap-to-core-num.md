---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: grid 总数不超过核数
description: grid = (min(total_blocks, num_cores),)，禁止直接 grid=(total_blocks,) 导致超核数空跑
tags: [grid_cap, num_cores, no_oversubscribe]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-2 grid 总数不超过核数

【优化维度】：计算｜多核负载
【泛化层级】：Triton通用（迁移条件：输出规模大、total_blocks 可能超核数；失效边界：total_blocks <= num_cores 时可用 2D grid 充分利用多核）

<原则：输出规模大时 total_blocks 可能远超核数（如上采样 N*C*H_out），超 grid 上限且多余 program 空跑。必须钳制到核数。核数读取见 [[dynamic-vector-core-num]]，负载均衡见 [[load-balance-formula]]。>

<骨架（占位名伪码）：
grid = (min(total_blocks, num_cores),)   # 钳制到核数
# kernel 内交错循环处理多个 block
for blk in range(pid, total_blocks, num_cores):
    ...
>

**坏实践（反例）**：grid 直接等于 total_blocks
```python
grid = (total_blocks,)   # total_blocks 可能 = 10000，远超核数
```
后果：超核数的 program 空跑，调度开销大；可能超 grid 上限报错。✅ 正解即 min(total_blocks, num_cores)。

**已知实例**：tensor-transform G4（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 同属优化知识，主题相关
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 主题相关：共享 grid_cap, num_cores
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 主题相关：共享 no_oversubscribe
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 主题相关：共享 num_cores
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
