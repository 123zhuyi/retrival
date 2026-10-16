---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fused state update"
description: "conv 场景下，Memory Access Optimization: fused state update，收益 35%"
tags: [访存效率, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fused state update

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，KERNEL_WIDTH 与 seqlen 为编译期常量的短序列 decode 场景；失效边界：seqlen 无法在编译期确定或序列较长时，全展开与寄存器窗口方案退化为通用循环，收益消失；KERNEL_WIDTH >8 时不宜全展开）

### 原则
- **fused state update**：The kernel fuses the sliding window state update (shifting the convolution buffer) with the convolution computation and activation (SiLU). This eliminates the need for a separate kernel to manage the recurrent state, significantly reducing memory access overhead and kernel launch latency.
- **vectorized layout transformation**：Utilizes a specific sequence of reshape and transpose operations (e.g., `reshape(sub_align_dim, align_val * state_len).trans()`) to transform the data layout. This ensures that the `insert_slice` operation (concatenation of history and new tokens) writes to contiguous memory addresses, optimizing for the hardware's burst memory access capabilities.
- **channel blocking**：Processes the channel dimension (`dim`) in blocks of `DIM_BLOCK`. This tiling strategy improves data locality by keeping weights and state slices in registers or cache during the computation loop, reducing global memory fetches.
- **mixed precision accumulation**：Performs the convolution accumulation in float32 (`x_new_T.to(tl.float32)`) to maintain numerical stability for the recurrent state, which is critical for preventing gradient explosion or vanishing in long sequences, while storing the result in the input precision.
- **sliding window register reuse**：The kernel loads the previous 'KERNEL_WIDTH-1' tokens from the convolution state into registers (col0, col1, etc.) before the main computation loop. Inside the loop, it shifts these registers and loads only the current new token. This drastically reduces global memory bandwidth usage by avoiding redundant loads of historical data for each time step.
- **vectorized state update**：The update of the circular buffer (conv_state) is performed by loading a block of history (blockM), shifting the data in local memory/registers to accommodate new tokens, and then storing the entire block back. This is significantly more efficient than scalar or per-token updates to the global state buffer.
- **static loop unrolling**：The inner loop over the kernel width utilizes `tl.static_range` and compile-time constants (`KERNEL_WIDTH`). This allows the compiler to fully unroll the convolution accumulation logic for specific widths (2, 3, or 4), eliminating runtime branching overhead and enabling efficient instruction pipelining.
- **state stitching strategy**：For the initial time tile (ti==0), the kernel efficiently loads the tail of the existing convolution state and stitches it with the head of the new input using tl.insert_slice. This avoids redundant memory loads of historical data and maintains continuity for the sliding window operation.
- **grid stride dynamic scheduling**：Utilizes a 1D program grid with a grid-stride loop (tl.range) to map 3D tasks (Batch, Dim, Time). This maximizes hardware occupancy and load balancing, particularly beneficial when tensor dimensions are not perfectly divisible by block sizes.
- **block ptr vectorization**：Employs tl.make_block_ptr for loading weights and storing outputs. This facilitates efficient 2D memory access patterns and leverages hardware vectorization units, reducing the number of memory instructions compared to scalar pointer arithmetic.
- **optimized state update**：Implements a specialized path for updating the convolution state (conv_state_update_ptr) that handles negative offsets via a pre-defined head tile size (ST_STORE_HEAD_TILE_SIZE). This avoids expensive dynamic branching or tl.maximum calls in the critical store path.
- **tiling strategy**：Implements 2D tiling using B_TILE (batch dimension) and blockN (channel dimension). This groups multiple sequences and channels into a single kernel instance, significantly improving data locality and enabling weight reuse across the batch tile.
- **vectorized state update**：Replaces scalar operations for state management with a chunked copy mechanism (T_CHUNK). It efficiently handles the sliding window update (shifting old history and appending new tokens) using vectorized loads/stores and masks, which is critical for minimizing latency in state-space models (SSMs).
- **weight preloading**：Loads convolution weights and bias into registers once per program instance. These values are persistently reused across all tokens in the sequence and all batches within the tile, drastically reducing global memory bandwidth consumption.
- **mixed precision compute**：Utilizes FP16 for storing weights, states, and inputs to minimize memory footprint and transfer overhead, while performing the accumulation in FP32 to preserve numerical accuracy during the reduction operations.
- **data layout alignment**：The kernel performs a specific sequence of reshape and transpose operations (e.g., reshape(sub_align_dim, align_val * X).trans()) on the input, state, and weight tensors. This transforms the data layout to match the hardware's vectorized memory access granularity (implied by align_val), ensuring that memory loads/stores are coalesced and utilize the full bus width.
- **operator fusion**：The kernel fuses multiple stages of the causal convolution update into a single launch loading the previous state, concatenating it with new input, performing the convolution computation, applying bias and SiLU activation, and finally slicing and storing the updated state back to global memory. This significantly reduces global memory read/write bandwidth requirements.
- **continuous batching support**：Efficiently handles continuous batching by using an indirection array (conv_state_indices_ptr) to map program IDs to actual batch indices. This allows the kernel to process dynamic batches without the overhead of padding or reordering data in global memory.
- **register blocking and vectorization**：The loop over the dimension is tiled using DIM_BLOCK, and data is processed in chunks that fit into registers. The layout transformation enables the use of vectorized instructions for the convolution arithmetic (element-wise multiplication and reduction), improving compute throughput.
- **register tiling sliding window**：The kernel pre-loads the historical state tokens (col0, col1, etc.) and convolution weights into registers before entering the main token loop. Inside the loop, it only loads the current input token and updates the register-based window, significantly reducing global memory access frequency for the history data.
- **block wise circular buffer**：Instead of shifting the entire convolution state tensor one step at a time, the kernel loads a chunk of the state (defined by NP2_STATELEN and blockN), shifts the data in shared memory or registers, and writes it back. This block-wise approach maximizes memory bandwidth utilization for the state update step.
- **spec decoding adaptation**：The implementation includes specific logic for Speculative Decoding by handling `conv_state_token_offset` derived from `num_accepted_tokens`. This allows the kernel to correctly update the state buffer position based on variable acceptance rates without re-computing the entire history.
- **static loop unrolling**：The use of `tl.static_range` for the inner loop over `KERNEL_WIDTH` and the outer loop over `seqlen` allows the compiler to fully unroll loops and generate optimized instruction sequences for the small, fixed-size operations typical in decoding phases.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/188-_causal_conv1d_update_kernel.py
```

**坏实践（反例）**：卷积计算与滑窗状态更新拆成两个 kernel（或 kernel 内先算完再单独搬运 state）
```python
# kernel 1: 只做卷积 + 激活
# kernel 2: 读旧 state → 移位 → 追加新 token → 写回 state
```
后果：多一次 kernel launch 延迟，且 state 缓冲区多一轮 GM 读写；短序列 decode 场景这部分开销占比极高。✅ 正解：状态移位/拼接与卷积、激活融合在单次 launch 内，窗口末尾的 W-1 个值直接复用为待写回的新 state。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_001.yaml`)
- 声称性能收益：35%
- 备注：适用于自回归 decode 阶段（序列短）的因果卷积增量更新场景。需 KERNEL_WIDTH 与 seqlen 为编译期常量以获得最大性能

<!-- okf:related:start -->

# 相关

- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
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
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
