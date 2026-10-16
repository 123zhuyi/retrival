---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: overlapping t tiling"
description: "conv 场景下，Memory Access Optimization: overlapping t tiling，收益 减少约 (num_T_tiles - 1) × (W-1) × D_CHK_SIZE 次全局内存 load，典型场景节"
tags: [访存效率, convolution-operations, conv]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: overlapping t tiling

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，序列需沿 T 维切 tile 的滑窗类 kernel；失效边界：W 较大时重叠 buffer（T_CHK+W-1）的 UB 开销占比升高，需重算 UB 预算；单 program 处理整条序列（无 T-tiling）的场景不适用）

### 原则
- **overlapping t tiling**：每个 T-tile 向前多加载 (W-1) 个 token，使 x_b 的 buffer 长度 = T_CHK_SIZE + W - 1。T-tile 0 从 conv_state 取 W-1 个历史 token 与当前输入拼接；后续 tile (ti>0) 从 x 的 (ti*T_CHK_SIZE - (W-1)) 位置开始加载，前 W-1 列自然等于上一个 tile 的最后 W-1 个 token。这保证了卷积窗口在 tile 边界处的连续性，无需每个 tile 都从 conv_state 读取历史。
- **redundant load elimination**：相比 naive 方案（每个 tile 独立从 conv_state 取 W-1 个历史 token），重叠 T-tiling 消除了后续 tile 的 (W-1) × D_CHK_SIZE 次额外全局内存访问。对于 seq_len=1024, T_CHK_SIZE=256, W=4 的场景，共 4 个 tile，节省了 3×(W-1)×D_CHK_SIZE = 9×D_CHK_SIZE 次 load。
- **state stitching first tile**：仅第一个 T-tile (ti==0) 需要从 conv_state 加载历史 token。使用 `tl.insert_slice` 将当前输入 x 插入到加载的状态 buffer 的第 (W-1) 列之后，实现零拷贝拼接。`tl.load(conv_state_ptr, offsets=(di*D_CHK, state_len-(W-1)), block_shape=(D_CHK, (W-1)+T_CHK))` 一次加载状态尾部和当前 tile 所需空间。
- **buffer size tradeoff**：x_b buffer = T_CHK_SIZE + W - 1 个元素（而非 T_CHK_SIZE）。额外 (W-1) 列作为重叠区，代价是 UB 用量略微增加，收益是消除了跨 tile 的冗余历史加载。对于 W=4 的典型场景，buffer 仅增加约 1.2%（3/256），代价可忽略。
- **state update tile awareness**：状态更新（conv_state_update）需要判断当前 tile 是否覆盖了状态保存区 `[seq_len - state_len, seq_len)`。通过计算 `t_end_off = (ti+1)*T_CHK_SIZE` 与 `new_state_start_off = seq_len - state_len` 的重叠关系，仅在覆盖的 tile 中执行状态提取和写入。避免每个 tile 都做无效的状态更新检查。

### 代码/骨架出处
- `first_tile`:
```python
# T-tile 0: 从 conv_state 取历史 + 拼接当前输入
st_b = tl.load(conv_state_ptr + bi*state_len*dim,
               offsets=(di*D_CHK_SIZE, state_len - (width-1)),
               block_shape=(D_CHK_SIZE, (width-1) + T_CHK_SIZE))
x_b_tmp = tl.load(x_ptr + bi*dim*seq_len,
                  offsets=(di*D_CHK_SIZE, ti*T_CHK_SIZE),
                  block_shape=(D_CHK_SIZE, T_CHK_SIZE))
x_b = tl.insert_slice(st_b, x_b_tmp, (0, width-1), (D_CHK_SIZE, T_CHK_SIZE))
```
- `subsequent_tile`:
```python
# T-tile i (i>0): 向前偏移 W-1，加载重叠区
x_b = tl.load(x_ptr + bi*dim*seq_len,
              offsets=(di*D_CHK_SIZE, ti*T_CHK_SIZE - (width-1)),
              block_shape=(D_CHK_SIZE, T_CHK_SIZE + width - 1))
```

**坏实践（反例）**：naive 方案——每个 T-tile 都独立从 conv_state 读 W-1 个历史 token 再拼当前输入
```python
for ti in range(num_T_tiles):
    hist = tl.load(conv_state_ptr + ...)   # 每个 tile 重复读历史（ti>0 时纯属冗余）
    x_b  = tl.load(x_ptr + ti*T_CHK, ...)
```
后果：后续 tile 各多 (W-1)×D_CHK_SIZE 次 GM load；seq_len=1024、T_CHK=256、W=4 时多 9×D_CHK_SIZE 次。✅ 正解：重叠 T-tiling——x_b buffer 扩为 T_CHK+W-1，ti>0 的 tile 从 `ti*T_CHK-(W-1)` 起载，前 W-1 列天然等于上一 tile 尾部，历史 load 归零（仅 ti==0 读一次 conv_state）。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_004.yaml`)
- 声称性能收益：减少约 (num_T_tiles - 1) × (W-1) × D_CHK_SIZE 次全局内存 load，典型场景节省 15-25% 内存带宽
- 备注：重叠 T-tiling 是 Update kernel 在 BDT 布局下实现高效 tile 并行化的核心设计。与 Prefill kernel（一个 program 处理整条序列，无需 T-tiling）形成对比。该模式是序列处理 kernel 做 T 维度切分时的通用范式。

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](../compute-pipeline/exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](../compute-pipeline/exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](../compute-pipeline/exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](../compute-pipeline/exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
