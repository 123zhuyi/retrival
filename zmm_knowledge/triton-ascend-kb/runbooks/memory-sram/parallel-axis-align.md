---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 并行轴选择必须与连续访问对齐，而非与最大 numel 对齐
description: 按算子语义选并行轴使内层访问沿 trailing/E/D 连续维度，否则离散访问和原子竞争性能跌至 0.1x
tags: [parallel_axis, contiguous_access, gather_scatter, axis_select]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-8 并行轴与连续访问对齐

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：gather/scatter/索引类算子；失效边界：纯 element-wise 无索引）

<原则：索引算子核心计算量低于访存/重排，并行轴若不与连续访问对齐会引入离散全局访问和原子竞争，性能通常跌至 0.1x 以下。Gather 类优先沿 outer 和 index_len 并行、沿 trailing 连续读；Scatter 类按 trailing 宽度选输出位置并行或 index 分块向量 atomic_add。与 [[vectorized-coalesced-io]] 配套。>

<骨架（占位名伪码）：
# Gather: 每个 program 处理一段 (o, k)，沿 trailing 连续读
pid = tl.program_id(0)
o_idx = pid // index_len
k_idx = pid % index_len
# 沿 trailing 连续 load/store
vals = tl.load(src_ptr + base + tl.arange(0, BLOCK_T), mask=mask)
>

**坏实践（反例）**：按最大 numel 选并行轴但内层离散
```python
# 并行轴选 index 但 trailing 短，内层 gather 发散地址
```
后果：离散全局访问，性能 0.1x 以下。✅ 正解即并行轴对齐连续访问。

**已知实例**：index-computation G1（Gather/Scatter 并行轴选择）

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 主题相关：共享 contiguous_access
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 contiguous_access
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 contiguous_access
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 contiguous_access
- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter
- 相关主题: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](pergroup-blockk-align.md) — 同属优化知识，主题相关
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](row-width-512b-align.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
