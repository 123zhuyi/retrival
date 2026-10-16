---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: numerical stability reduction'
description: 'loss 场景下，Compute Optimization: numerical stability reduction，收益 25%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, loss]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: numerical stability reduction

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：num_tokens < 128 时原子加与块归并的固定开销占比高；split 数为 1 时跨 split 的稳定归并无意义；eps=1e-12 阈值按 fp32 设定，更低精度需复核）

### 原则
- **numerical stability reduction**：The kernel implements a numerically stable reduction algorithm (a variant of Log-Sum-Exp) to aggregate results across splits. It tracks the global maximum value and dynamically scales the accumulation terms (`global_accu` and `global_entropy_b`) using exponential differences (`exp(old_max - new_max)` and `exp(local_max - new_max)`). This prevents floating-point overflow or underflow that is common when summing probabilities or log-probabilities over large sequences.
- **vectorized block loading**：Data is loaded in 2D blocks defined by `BLOCK_SIZE_M` and `BLOCK_SIZE_N` (32x64) instead of scalar operations. This approach maximizes memory coalescing and utilizes the GPU's memory bandwidth more efficiently by fetching multiple intermediate values (max, accu, entropy_b) per transaction.
- **fused computation pipeline**：The kernel fuses multiple stages of the entropy calculation into a single pass reducing intermediate statistics, calculating the final entropy value (`log(accu) + max - entropy_b/accu`), and updating the log probabilities. This fusion minimizes global memory read/write overhead and kernel launch latency compared to executing these operations in separate kernels.
- **atomic scalar aggregation**：For the final reduction step (Sum or Mean), the kernel utilizes `tl.atomic_add` to aggregate results from different program IDs into a scalar pointer. This allows the parallel computation of per-token logprobs while safely combining them into the final global loss metric without race conditions.
- **tiled reduction strategy**：The kernel iterates over the split dimension (`num_splits`) using a block size of 128 (`BLOCK_SIZE_N`), processing 32 tokens (`BLOCK_SIZE_M`) simultaneously. This tiling strategy balances register pressure and memory bandwidth, significantly reducing the number of global memory accesses compared to a scalar reduction.
- **numerically stable logsumexp**：Implements the LogSumExp algorithm to ensure numerical stability when summing exponentials. It dynamically tracks the `global_max` and rescales the accumulated sum (`global_accu`) and entropy term (`global_entropy_b`) using coefficients `_coeff` (for previous accumulator) and `_scale` (for current chunk), preventing overflow/underflow.
- **vectorized memory access**：Utilizes 2D pointer arithmetic with broadcasting (`offs_m[:, None]` and `offs_n[None, :]`) to load tiles of data into registers. This allows the hardware to coalesce memory reads more effectively than iterating through individual elements.
- **numerical stability**：Implements an epsilon threshold (eps=1e-12) applied to the accumulate term using tl.maximum before division and logarithm operations. This prevents NaN generation during division by zero or log(0) calculations, which is critical for entropy and logprob computations.
- **vectorized memory access**：Utilizes a block size of 128 (BLOCK_SIZE_M) to process multiple tokens simultaneously. This ensures memory loads and stores for logprobs, maximums, and accumulates are coalesced, significantly improving memory bandwidth utilization compared to scalar operations.
- **kernel fusion**：Fuses the entropy calculation, log-probability adjustment (NLL computation), and the final reduction step (sum/mean) into a single kernel. This eliminates the need for writing intermediate results back to global memory and reduces kernel launch overhead.
- **atomic reduction**：Leverages tl.atomic_add for the global reduction of the scalar loss (logprobs_scalar). This allows multiple thread blocks to compute partial reductions in parallel without requiring explicit inter-block synchronization or a separate reduction kernel.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/203-efficient_entropy_triton_epilogue_tp_update.py
```

**坏实践（反例）**：跨 split 直接对概率/对数概率累加而不做 max 跟踪与 rescale → 长序列大数值下 exp 上溢/下溢，结果出 NaN → ✅ 正解即本条：全局 max 跟踪 + `exp(old_max - new_max)` rescale 归并，除法/log 前 `tl.maximum(accu, 1e-12)` 防 0 除。

**已知实例**：loss §raw (`experiences/loss/exp_entropy_epilogue_001.yaml`)
- 声称性能收益：25%
- 备注：The kernel supports dynamic reduction modes (store, sum, mean) via runtime branching. For optimal performance, ensure num_tokens is sufficiently large to hide the latency of the atomic operations.

<!-- okf:related:start -->

# 相关

- 开发指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 loss
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, loss
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
