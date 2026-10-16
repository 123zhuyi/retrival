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

**已知实例（段连续 gather 发射 floor + 接口不可用记录）**：grid_sample（gather_scatter，ascend910b/Ascend910_9372，CANN 9.1.0）——NHWC 段连续 gather（64B 段）实测为**发射路径 floor**：simulator MTE2 80.5%、~278cyc/段（≈98ns），有效带宽 31GB/s << 213GB/s microbench（非带宽饱和）；128B 成对段合并使发射数减半，但 UB（192KB/核）迫使 BHW 256→128 净零抵消（10.266ms vs 10.237ms）；`gather_out_to_ub` 硬件 gather 接口在本栈 bishengir-compile 失败（文档明示降级普通接口）。段数为采样语义下限（bilinear 4 tap/点）时无进一步优化点（2026-09-17；证据：op_12_12_grid_sample_20260917_1453_7647/output/opt_iter_2/log.md（simulator 覆盖表）+ opt_iter_1_ir_3/log.md（成对段消融））

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 主题相关：共享 binary_search
- 实践案例: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](../../ops/moe-init-routing-v2.md) — 主题相关：共享 gather
- 实践案例: [MoeInitRouting (MoE routing) on Ascend NPU](../../ops/moe-init-routing.md) — 主题相关：共享 gather
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 gather
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../compute-pipeline/scalar-load-channel-index.md) — 同属优化知识，主题相关
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](../compute-pipeline/work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
