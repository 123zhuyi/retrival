---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: gather_scatter
dimension: "2.3"
title: Radix Sort Scatter 的 Lookback 调度优化
description: 针对 radix sort scatter 阶段，通过 lookback 前缀和、volatile 同步、cache-global 写回与多 bin 数据复用，把全局前缀和合并到同一个 kernel，提升 25%。
tags: [gather_scatter, scheduling, radix_sort, lookback, volatile, cache_global, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/gather_scatter/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Radix Sort Scatter 的 Lookback 调度优化

## 概述

Radix sort 的 scatter 阶段需要把每个元素按当前 digit 的桶号写到全局输出中的正确位置。传统做法是先用一个全局 prefix-sum kernel 计算每个 block 在每个桶中的偏移，再启动 scatter kernel。该经验通过 **lookback 机制** 把这两步合并到一个 kernel 中：每个 block 先计算本地直方图，然后向前查询其他 block 的状态，逐步累加得到全局偏移。

核心权衡：

- 合并 kernel 减少了内存往返和 launch 开销；
- lookback 引入跨 block 同步，需要使用 `volatile=True` 保证弱内存模型下的正确性；
- `cache_modifier=".cg"` 让同步数据尽快写到 L2/主存，避免其他 block 读到过期值；
- 单个 block 内遍历多个 bin，复用一次性加载的 `key_data`/`value_data`。

收益在 `n_elements ≥ 1024`、`bins ≥ 2` 时约 25%，适合大规模 radix sort / randperm 类算子。

## 共同模式

### 1. Lookback 前缀和替代全局扫描

每个 block 计算自己负责的 `BLOCK_SIZE` 个元素在当前 digit 上的局部分布，然后向前逐个查询 lookback buffer。如果前一个 block 已经写出 **global** 计数，则可以直接累加并停止；如果只是 **partial** 计数，则继续向前查询。这样无需单独的全局 prefix-sum pass。

### 2. Volatile load 保证同步正确性

在 spin-wait 循环中，`tl.load(d_lookback + ..., volatile=True)` 绕过 L1 cache，直接从 L2/主存读取最新值。在 Ascend/GPU 等弱一致内存模型下，这是避免死锁和读到旧值的关键。

### 3. Cache-global 写回让同步数据立即可见

对 lookback buffer 的 store 使用 `cache_modifier=".cg"`，提示编译器把数据直接写向 global memory/L2，而不是留在 L1。这样后续 block 的 volatile load 能尽快看到更新。

### 4. 单 block 多 bin 复用输入数据

`for s in range(bins_segment)` 让同一个 block 连续处理多个 bin。`key_data` 和 `value_data` 只在 kernel 开头加载一次，随后在所有 bin 的处理中复用，显著降低 HBM 带宽。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_radix_sort_scatter_001` | 25% | Radix sort scatter 通过 lookback 前缀和、volatile 同步与 cache-global 写回合并全局扫描与 scatter，提升 25%。 |

## 代表 Kernel

以下为 `exp_radix_sort_scatter_001` 对应的 kernel：

```python
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("randperm"),
    key=["n_elements"],
)
@triton.jit
def radix_sortbykey_scatter_kernel(
    key_out,
    value_out,
    key_in,
    value_in,
    digit_hist,
    d_lookback,
    n_elements,
    bit_offset,
    passes,
    p,
    num_portions,
    portion_size,
    portion_id,
    bit_mask,
    bins_segment,
    max_tiles_per_portion,
    bins: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    LOOKBACK_PARTIAL_MASK = 1 << 30
    LOOKBACK_GLOBAL_MASK = 1 << 31
    LOOKBACK_KIND_MASK = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK
    LOOKBACK_VALUE_MASK = ~LOOKBACK_KIND_MASK

    pid0 = tl.program_id(0)
    portion_id_i64 = portion_id
    portion_id_i64 = portion_id_i64.to(tl.int64)
    key_offset = (
        portion_id_i64 * portion_size
        + pid0.to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    key_mask = key_offset < n_elements
    value_data = tl.load(value_in + key_offset, mask=key_mask)
    key_data = tl.load(key_in + key_offset, mask=key_mask)

    ikey_data = radix_type_convert(key_data)
    key_digit = (ikey_data >> bit_offset) & bit_mask

    blk_bin_start = tl.program_id(1) * bins_segment
    last_block = tl.program_id(0) == tl.num_programs(0) - 1
    for s in range(bins_segment):
        bin_id = s + blk_bin_start
        key_digit_mask = (key_digit == bin_id) & key_mask
        key_elem_mask = tl.where(key_digit_mask, 1, 0)
        key_block_rank = tl.cumsum(key_elem_mask)
        key_block_rank = tl.where(key_digit_mask, key_block_rank - 1, 0)
        bin_of_bucket = tl.sum(key_elem_mask)
        partial_counter = bin_of_bucket | LOOKBACK_PARTIAL_MASK
        tl.store(
            d_lookback
            + ((portion_id * passes + p) * max_tiles_per_portion + pid0) * bins
            + bin_id,
            partial_counter,
            cache_modifier=".cg",
        )
        bin_offset = p * (bins + 1) + bin_id
        prefix_offsets = tl.load(
            digit_hist + bin_offset + portion_id * passes * (bins + 1)
        )
        bk = pid0 - 1

        inc_sum = bin_of_bucket

        while bk >= 0:
            rd_lbk_offset = (
                (portion_id * passes + p) * max_tiles_per_portion + bk
            ) * bins + bin_id
            partial_prefix = tl.load(d_lookback + rd_lbk_offset, volatile=True)
            max_wait = 1000
            wait_count = 0
            while partial_prefix == 0 and wait_count < max_wait:
                partial_prefix = tl.load(d_lookback + rd_lbk_offset, volatile=True)
                wait_count += 1
            inc_sum += (partial_prefix & LOOKBACK_VALUE_MASK).to(tl.int32)
            if partial_prefix & LOOKBACK_GLOBAL_MASK:
                bk = -1
            else:
                bk -= 1

        global_counter = inc_sum | LOOKBACK_GLOBAL_MASK
        tl.store(
            d_lookback
            + ((portion_id * passes + p) * max_tiles_per_portion + pid0) * bins
            + bin_id,
            global_counter,
            cache_modifier=".cg",
        )
        inc_bucket_offset = prefix_offsets.to(tl.int64) + inc_sum.to(tl.int64)
        if last_block and portion_id < num_portions - 1:
            tl.store(
                digit_hist + bin_offset + (portion_id + 1) * passes * (bins + 1),
                inc_bucket_offset,
            )
        global_offsets = (
            inc_bucket_offset - bin_of_bucket.to(tl.int64) + key_block_rank.to(tl.int64)
        )
        tl.store(key_out + global_offsets, key_data, mask=key_digit_mask)
        tl.store(value_out + global_offsets, value_data, mask=key_digit_mask)
```

## 已知实例

- `exp_radix_sort_scatter_001` — Scheduling Optimization（收益 25%）
  - 适用于 radix sort / randperm scatter 阶段，关键参数 `n_elements ≥ 1024`、`bins ≥ 2`。
  - 对应 kernel: `raw_experiences/extracted_kernels/137-radix_sortbykey_scatter_kernel.py`

## 关联知识

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, gather_scatter
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 同属优化知识，主题相关
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 同属优化知识，主题相关
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](../../ops/moe-init-routing-v2.md) — 同属优化知识，主题相关
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](../../ops/moe-init-routing-v3.md) — 同属优化知识，主题相关
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](../../ops/moe-init-routing.md) — 同属优化知识，主题相关
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 同属优化知识，主题相关
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../memory-sram/parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, gather_scatter
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, gather_scatter
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
