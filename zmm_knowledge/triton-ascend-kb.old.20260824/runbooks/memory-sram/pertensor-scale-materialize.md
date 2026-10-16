---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load
description: pertensor scale=[1] 经 expand 后必须 contiguous 物化为 [1,N]，stride-0 视图会触发 UB overflow
tags: [pertensor_scale, materialize, stride_zero, ub_overflow, broadcast_load]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-7 pertensor scale 物化连续

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：pertensor 量化 scale=[1] 需广播到 [1,N]；失效边界：pergroup/perchannel scale 已是连续张量）

<原则：pertensor scale=[1] 经 expand 得 stride-0 视图，Ascend 编译器无法 tile，触发 UB overflow（实测 790KB > 192KB UB）。必须 contiguous 物化为 [1,N] 走正常向量 load。>

<骨架（占位名伪码）：
# pertensor scale=[1] -> 物化为连续 [1,N]
scale_mat = scale.view(1, 1).expand(1, N).contiguous()   # stride (N,1)，连续
# kernel 内正常向量 load
s = tl.load(scale_ptr + offs_n)
>

**坏实践（反例）**：stride-0 广播 load
```python
scale_mat = scale.view(1, 1).expand(1, N)   # stride-0 视图，未 contiguous
# kernel 内 tl.load(scale_ptr + offs_n * 0)  # 编译器无法 tile
```
后果：UB overflow（790KB > 192KB），iter_0 教训。✅ 正解即 contiguous 物化。

**已知实例**：weight-quant-matmul W3（iter_0 UB overflow 教训）

<!-- okf:related:start -->

# 相关

- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](../compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../compute-pipeline/scalar-load-channel-index.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](../compute-pipeline/work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 ub_overflow
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
