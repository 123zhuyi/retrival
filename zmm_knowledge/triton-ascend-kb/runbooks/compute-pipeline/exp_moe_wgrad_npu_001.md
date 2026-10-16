---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: memory alignment'
description: 'moe 场景下，Compute Optimization: memory alignment，收益 25%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, moe]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: memory alignment

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：BLOCK_X 必须是硬件向量宽度倍数（备注自述为性能关键）；`tl.multiple_of` 的对齐承诺必须与真实分配一致，错误承诺导致越界或异常；NUM_COLUMNS < 128 时向量化配置需重调）

### 原则
- **memory alignment**：The use of `tl.multiple_of` for pointer arithmetic ensures that memory addresses adhere to the strict alignment requirements of the NPU (e.g., 32-byte boundaries), which is critical for enabling high-efficiency memory load/store instructions and avoiding hardware exceptions or penalties.
- **precision promotion**：Intermediate accumulation is performed in `tl.float32` (`acc`), while inputs are cast from lower precision (`fp16`/`bf16`). This prevents numerical precision loss and overflow during the reduction of large vectors, which is a common issue in mixed-precision training.
- **ilp batching**：The kernel processes multiple tokens (`blockSize`) per program ID within the `for idx_in_block` loop. This increases Instruction Level Parallelism (ILP), effectively hiding memory latency by overlapping the computation of one token with the memory loads of another.
- **vectorized loads**：The inner loop iterates with a step of `BLOCK_X`, loading chunks of the vector. Tuning `BLOCK_X` to match the NPU's vector register width (e.g., 128 bytes) ensures that each load instruction maximizes the data bus utilization.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/004-_binned_copy_wgrad_npu.py
```

**坏实践（反例）**：大向量归约直接用 fp16/bf16 累加、逐 token 串行处理 → 精度丢失 + 访存延迟无掩盖 → ✅ 正解即本条：fp32 累加 + 每 program 批量处理 blockSize 个 token（ILP）+ `tl.multiple_of` 对齐提示。

**已知实例**：moe §raw (`experiences/moe/exp_moe_wgrad_npu_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically designed for the sparse weight gradient calculation in Mixture of Experts (MoE) layers. The performance relies heavily on `BLOCK_X` being a multiple of the hardware's native vector width to ensure full vectorization.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../memory-sram/exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 同属优化知识，主题相关
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
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
