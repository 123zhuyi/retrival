---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: software pipelining"
description: "gather_scatter 场景下，Memory Access Optimization: software pipelining，收益 35%"
tags: [访存效率, indexing-operations, gather_scatter]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: software pipelining

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp8/fp16/bf16，HIDDEN_SIZE≥128、topk_num≥1；失效边界：HIDDEN_SIZE<128 时整行向量加载退化；无 FP8 scale 的场景 SCALE 相关 padding 路径多余；多个核并发写同一 expert 且未用 grid-stride 分摊时，`expert_start_loc` 更新存在竞争风险，需同步保护）

### 原则
- **software pipelining**：The loop `tl.range(..., num_stages=4)` employs software pipelining to prefetch memory loads and overlap them with store operations, effectively hiding memory latency during the scatter phase.
- **vectorized scatter**：Instead of processing tokens one by one, the kernel writes the expert ID to the `m_indices` buffer in chunks of `BLOCK_E`, significantly improving memory bandwidth utilization.
- **local prefix sum**：The kernel calculates the starting offsets (`expert_start_loc`) for experts using `tl.cumsum` on loaded blocks of token counts, allowing for parallel offset computation without expensive global synchronization.
- **vectorized memory access**：The kernel utilizes `tl.arange` to load and store the entire `HIDDEN_SIZE` vector in a single operation. This maximizes memory bus utilization and reduces the number of memory instructions issued compared to element-wise access.
- **memory alignment padding**：By using `HIDDEN_SIZE_PAD` and `SCALE_HIDDEN_SIZE_PAD`, the kernel ensures that all memory accesses are aligned to the hardware's burst size (e.g., 32 or 128 bytes). This avoids unaligned access penalties and is crucial for achieving peak bandwidth on Ascend hardware.
- **input data reuse**：The input token data (`recv_x`) and its scale (`recv_x_scale`) are loaded once per token before the inner loop. This data is then reused for all `topk_num` experts, minimizing redundant global memory reads when tokens are routed to multiple experts.
- **software pipelining**：The inner loop uses `num_stages=4` to prefetch expert indices and destination pointers. This hides the latency of indirect memory loads (pointer chasing) required to determine the write location for each expert.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/061-_fwd_kernel_ep_scatter_2.py
```

**坏实践（反例）**：逐 token 标量写路由索引、每个目标 expert 重读同一输入，且无流水线掩盖间接寻址
```python
for k in range(topk_num):
    x = tl.load(recv_x_ptr + row_off)            # 每 expert 重复读同一 token
    dst = tl.load(expert_start_loc_ptr + eid)    # 间接寻址，延迟无掩盖
    tl.store(m_indices_ptr + dst, eid)           # 逐标量写 expert ID
    tl.store(out_ptr + dst * HIDDEN_SIZE + ..., x)
```
后果：输入被读 topk_num 次、expert ID 写事务数等于 token×topk_num，pointer chasing 延迟完全暴露，35% 收益流失。✅ 正解：输入 token 及其 scale 每 token 载入一次、对全部 topk_num 个 expert 复用；`m_indices` 按 BLOCK_E 块写入；`tl.range(..., num_stages=4)` 软件流水预取 expert 索引与目标指针，并用 `tl.cumsum` 本地前缀和算 `expert_start_loc` 避免全局同步。

**已知实例**：gather_scatter §raw (`experiences/gather_scatter/exp_moe_scatter_001.yaml`)
- 声称性能收益：35%
- 备注：该 kernel 面向 token 分发到各 expert 的专家路由负载，支持 FP8 scaling。需确保 `expert_start_loc` 更新正确——若多个核并发写同一 expert 会有竞争风险，grid-stride 循环有助于分摊负载。

<!-- okf:related:start -->

# 相关

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 gather_scatter
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 gather_scatter, indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: software pipelining](../compute-pipeline/exp_swiglu_bwd_flat_001.md) — 同属优化知识，主题相关
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 gather_scatter, indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 访存效率
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter

<!-- okf:related:end -->
