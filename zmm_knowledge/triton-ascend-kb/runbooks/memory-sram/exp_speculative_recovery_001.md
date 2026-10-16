---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: vectorized chunking"
description: "misc 场景下，Memory Access Optimization: vectorized chunking，收益 30%"
tags: [访存效率, reduction-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: vectorized chunking

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp32/bf16，vocab_size ≥1024、SUB_BLOCK ≥64；失效边界：vocab_size <1024 时单块处理开销已低，分块收益稀释；SUB_BLOCK × 同时加载的数据流条数超出 UB 容量时需重算分块；显式提供候选概率的路径不适用原地修改技巧）

### 原则
- **vectorized chunking**：The kernel utilizes a `SUB_BLOCK` loop stride to process the vocabulary dimension. By loading chunks of probability vectors (`draft_probs`, `target_probs`, `q`) into vector registers simultaneously, it maximizes memory bandwidth utilization and reduces the overhead of scalar loads.
- **in place buffer reuse**：In the `NO_DRAFT_PROBS` execution path, the kernel temporarily modifies the `target_probs_ptr` in-place (zeroing out the draft token's probability) to exclude it from the argmax search, and subsequently restores the original value. This technique avoids the allocation of a separate temporary buffer, saving memory capacity and bandwidth.
- **masked boundary handling**：The use of `mask=vocab_offset < vocab_size` within the vectorized load instructions allows the kernel to handle arbitrary vocabulary sizes and padded dimensions efficiently without introducing divergent branching logic inside the processing loop.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/170-sample_recovered_tokens_kernel.py
```

**坏实践（反例）**：逐元素标量 load 处理 vocab 维，并为排除候选 token 另开临时 buffer
```python
for v in range(vocab_size):                       # 逐元素标量 load
    tv = tl.load(target_probs_ptr + v)
    ...
tmp = tl.load(target_probs_ptr + offs)            # 另分配/拷贝临时 buffer
tl.store(tmp_ptr + draft_token, 0.0)              # 在副本上清零以排除该项
best = argmax(tmp)
```
后果：标量 load 带宽利用率低；临时 buffer 额外占用内存并多付一趟拷贝带宽。✅ 正解：按 `SUB_BLOCK` 步长分块向量 load 多路概率向量（mask `vocab_offset < vocab_size` 处理边界）；无候选概率路径下原地将 `target_probs` 中候选项清零、argmax 完成后恢复原值，零额外分配。

**已知实例**：misc §raw (`experiences/misc/exp_speculative_recovery_001.yaml`)
- 声称性能收益：30%
- 备注：This kernel is critical for the token recovery phase in speculative decoding. The `NO_DRAFT_PROBS` optimization is specifically tailored for n-gram based speculation where explicit draft probabilities are not available.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_grpo_loss_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_topp_filter_001.md) — 主题相关：共享 misc, reduction-operations
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, reduction-operations, 访存效率

<!-- okf:related:end -->
