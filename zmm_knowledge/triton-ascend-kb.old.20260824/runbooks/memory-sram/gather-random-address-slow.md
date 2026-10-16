---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行
description: 发散地址的向量 gather 退化为近似串行，二分查找类用 per-expert 标量二分+grid 并行反而快 5-8x
tags: [gather, random_address, binary_search, per_expert, grid_parallel]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-10 gather 随机地址反模式

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：二分查找/搜索类算法各 lane 地址发散；失效边界：连续/对齐地址的 gather 高效）

<原则：Ascend Vector Core 的向量 load 对连续/对齐地址高效，对发散随机地址退化为近似串行。二分/搜索类各 lane 的 mid 不同，向量 gather 比 per-expert 标量二分慢 5-8x。必须走 per-expert 标量二分 + grid 并行（一个 program 一个 expert/查询），数据小天然 cache 命中。>

<骨架（占位名伪码）：
# 禁止：向量 gather 二分
# mid_vec = (lo_vec + hi_vec) // 2
# vals = tl.load(ptr + mid_vec)  # 各 lane 地址发散，退化为串行

# 正确：per-expert 标量二分 + grid 并行
pid = tl.program_id(0)
expert = pid              # 一个 program 一个 expert
lo, hi = 0, n
while lo < hi:            # 标量二分
    mid = (lo + hi) // 2
    v = tl.load(ptr + mid)  # 标量 load，连续地址 cache 命中
    if v < target: lo = mid + 1
    else: hi = mid
# grid=(N,) 超 num_cores 时用 capped-grid 连续划分
>

**坏实践（反例）**：向量 gather 发散地址
```python
mid_vec = (lo_vec + hi_vec) // 2          # 各 lane mid 不同
vals = tl.load(ptr + mid_vec)             # 发散地址，近似串行
```
后果：E=64 向量批量二分比标量 per-expert 慢 5-8x。✅ 正解即 per-expert 标量 + grid 并行。

**已知实例**：index-computation G12（upper_bound/lower_bound/searchsorted）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 主题相关：共享 binary_search
- 实践案例: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](../../ops/moe-init-routing-v2.md) — 主题相关：共享 gather
- 实践案例: [MoeInitRouting (MoE routing) on Ascend NPU](../../ops/moe-init-routing.md) — 主题相关：共享 gather
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 gather
- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](../compute-pipeline/grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](../compute-pipeline/grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](../compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../compute-pipeline/scalar-load-channel-index.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](../compute-pipeline/traj-conv-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](../compute-pipeline/traj-solve_tril-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](../compute-pipeline/work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
