---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: score-tile reuse with online softmax"
description: "attention 场景下，Memory Access Optimization: score-tile reuse with online softmax，收益 200%"
tags: [访存效率, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: score-tile reuse with online softmax

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；bf16/fp16/fp8；HEAD_DIM≤128、BLOCK_M≥64、BLOCK_N≥64；失效边界：HEAD_DIM>128 时超出适用域，tile 常驻 SRAM 的预算需重估；BLOCK_M/BLOCK_N<64 时 cube 单元利用率与复用收益下降；序列长度远小于 block 时 boundary_check 开销占比升高）

### 原则
- **tiling strategy**：Implements a block-wise tiling approach (BLOCK_R for inputA rows, BLOCK_C for inputB/inputC columns) to load inputB and inputC tiles into SRAM once and reuse them across multiple inputA row blocks. This significantly reduces HBM (High Bandwidth Memory) access frequency compared to a naive implementation.
- **online softmax**：Utilizes the online softmax algorithm (iteratively updating max value `m` and sum of exp `l`) to compute attention probabilities and output incrementally. This avoids materializing the large N x N attention matrix in global memory, drastically lowering memory footprint and bandwidth consumption.
- **autotuning config**：Employs `@triton.autotune` with a comprehensive search space over block sizes (e.g., 128x128, 64x128) and multibuffer settings. This allows the kernel to dynamically select the optimal configuration that maximizes cube-unit utilization and minimizes latency for specific sequence lengths (S) and head dimensions (H).
- **software pipelining**：Leverages the `multibuffer` configuration option to overlap memory load operations (for inputB and inputC tiles) with compute operations (matrix multiplications). This effectively hides memory latency, keeping the compute units busy.
- **mixed precision computation**：Performs matrix multiplications (`tl.dot`) in `bf16` (LOW_TYPE) to maximize hardware throughput, while maintaining softmax statistics and accumulation in `fp32` (HIGH_TYPE) to preserve numerical stability during the exponential and reduction operations.
- **kernel fusion**：Fuses the score matrix multiplication, softmax, and weighted-value matrix multiplication into a single kernel loop. This eliminates the need to write intermediate scores (S) and softmax output (P) back to global memory.
- **block tiling**：Processes data in tiles defined by blockM, blockN, and BLOCK_D. This maximizes data reuse in the on-chip memory (SRAM/Cache) by keeping blocks of inputA, inputB, and inputC resident while performing multiple compute operations.
- **efficient boundary handling**：Utilizes tl.load with boundary_check and padding_option="zero". This efficiently handles sequence lengths that are not perfectly divisible by the block size without introducing expensive branching logic or complex index calculations inside the kernel loop.
- **mixed precision accumulation**：Casts the softmax probabilities (p) to the data type of the inputC tensor (k_T.dtype) before the final dot product. This allows the use of lower precision formats (like FP8) for the accumulation phase when fp8_v is enabled, reducing memory bandwidth while maintaining FP32 stability for the softmax statistics.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/253-_sdpa_infer_single_block.py
```

**坏实践（反例）**：把完整 N×N 评分矩阵物化到全局内存
```python
S = tl.dot(q_blk, k_blk)              # 评分块
tl.store(S_ptr + ..., S)              # 写回 GM
P = softmax(tl.load(S_ptr + ...))     # 再从 GM 读回做 softmax
tl.store(P_ptr + ..., P)              # 概率再写回 GM
out = tl.dot(tl.load(P_ptr + ...), v_blk)
```
后果：中间评分与概率矩阵在 GM 两写两读，带宽消耗随 N² 增长；分块复用+在线 softmax 融合后实测收益 200%。✅ 正解：inputB/inputC tile 一次载入 SRAM，在多个 inputA 行块间复用；循环内在线更新 running max（m_i）与归一化因子（l_i），评分与概率永不写回 GM；fp8_v 开启时 p 先 cast 到 inputC 的低精度类型再做值矩阵乘。

**已知实例**：attention §raw (`experiences/attention/exp_sdpa_flash_001.yaml`)
- 声称性能收益：200%
- 备注：This function represents the inner loop of an online-softmax-style attention kernel. It assumes that the input pointers (K_T_block_ptr, V_block_ptr) are advanced correctly by an outer loop. The fp8_v flag enables low-precision computation for the inputC projection.

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention
- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 attention
- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](../compute-pipeline/exp_fused_attention_001.md) — 主题相关：共享 attention
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](../compute-pipeline/exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax](../compute-pipeline/exp_sdpa_infer_ascend_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_online_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 主题相关：共享 attention, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
