---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: vectorized comparison"
description: "misc 场景下，Memory Access Optimization: vectorized comparison，收益 50%"
tags: [访存效率, indexing-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: vectorized comparison

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，int32/int64，vec_len ≥1、max_spec_len ≥1 的投机采样校验场景；失效边界：max_spec_len 极小（1~2）时标量循环开销本就低，向量化收益稀释；token 向量长度超出寄存器/UB 一次性容纳范围时需分块加载比较）

### 原则
- **vectorized comparison**：Replace the scalar loop `for i in range(num_tokens1)` with vectorized operations. Load chunks of the candidate and reference token sequences into vectors using `tl.load` with masks, then compare them using vectorized equality checks to identify the rejection point in parallel, significantly reducing loop overhead.
- **pointer arithmetic optimization**：Pre-calculate the base pointer for the output buffer (`output_token_ids_ptr + position * (max_spec_len + 1)`) outside the inner verification loop. Use vectorized stores to write the accepted tokens back to memory in a single operation rather than iterating and storing one by one.
- **register reuse**：Avoid using `tl.get_element` inside loops which extracts scalar values. Instead, keep the loaded token vectors in registers/scratchpad memory for the duration of the comparison to minimize memory access latency.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/167-rejection_greedy_sample_triton.py
```

**坏实践（反例）**：标量循环逐 token 比较，并用 `tl.get_element` 抽取标量、逐个 store
```python
for i in range(num_tokens):
    d = tl.get_element(cand_vec, i)        # 从向量中逐个抽标量
    t = tl.get_element(ref_vec, i)
    if d != t:                             # 串行分支定位拒绝点
        reject_pos = i
        break
    tl.store(out_ptr + position*(max_spec_len+1) + i, d)   # 逐元素写回
```
后果：标量循环与逐元素抽取/写回在核内向量执行模型下效率低，访存延迟无法隐藏，吞吐受限。✅ 正解：带 mask 的 `tl.load` 成块加载两组 token 向量并驻留寄存器，向量化等值比较并行定位拒绝点；输出基址 `output_token_ids_ptr + position*(max_spec_len+1)` 在校验循环外预计算，接受 token 用一次向量 store 写回。

**已知实例**：misc §raw (`experiences/misc/exp_spec_dec_verify_001.yaml`)
- 声称性能收益：50%
- 备注：The current implementation relies on scalar loops and `tl.get_element`, which is inefficient under the per-core vector execution model. Vectorizing the token comparison and memory operations is crucial for high throughput.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: 2D Block Tiling](../compute-pipeline/exp_embedding_backward_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: vectorized binary search](../compute-pipeline/exp_multinomial_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: lookback algorithm](../compute-pipeline/exp_radix_sort_scatter_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Compute Optimization: vectorized load](../compute-pipeline/exp_reject_sampler_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Scheduling Optimization: vectorized global load](../compute-pipeline/exp_rejection_sample_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: batched reduction replacement](../compute-pipeline/exp_scatter_wgrad_001.md) — 主题相关：共享 indexing-operations
- 相关主题: [Scheduling Optimization: block local sorting](../compute-pipeline/exp_shuffle_blockwise_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: padding strategy](../compute-pipeline/exp_sort_bitonic_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](../compute-pipeline/exp_top_p_sampling_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](../meta-skills/exp_bonus_renew_001.md) — 主题相关：共享 indexing-operations, misc
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — 主题相关：共享 indexing-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 indexing-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
