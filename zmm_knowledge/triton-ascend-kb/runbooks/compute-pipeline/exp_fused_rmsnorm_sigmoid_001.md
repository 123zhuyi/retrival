---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: operator fusion'
description: 'normalization 场景下，Compute Optimization: operator fusion，收益 35%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: operator fusion

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：假设输入是两段拼接结构且 DIM 为偶数（备注自述），非拼接布局不适用；n_rows < BLOCK 时 grid-stride 退化；fp32 归约是精度前提，省略会在归一化层累积误差）

### 原则
- **operator fusion**：Fuses the RMSNorm and Sigmoid operations into a single kernel launch. This optimization significantly reduces global memory access by reading the input tensor 'QQ' only once (conceptually split) and writing two outputs without storing intermediate results to HBM, thus lowering memory bandwidth pressure.
- **dimension splitting strategy**：Implements a logical split of the input tensor 'QQ' along the feature dimension using HALF_DIM. The first half of the data is processed through the RMSNorm pipeline (variance calculation, normalization, scaling), while the second half is processed through the Sigmoid pipeline. This is highly efficient for packed data structures in transformer models.
- **grid stride scheduling**：Utilizes a grid-stride loop pattern (`range(pid, NUM_BLOCKS, NUM_CORES)`) to distribute the workload across all available GPU cores. This ensures high utilization even when the number of rows is not perfectly divisible by the block size or when the grid dimension is small, preventing core idleness.
- **mixed precision computation**：Performs the reduction (sum of squares) and reciprocal square root operations in float32 (`q.to(tl.float32)`) to maintain numerical stability and prevent overflow/underflow, which is critical for normalization layers. The results are cast back to the input data type only at the final store stage to save memory bandwidth.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/083-fused_rmsnorm_and_sigmoid_kernel.py
```

**坏实践（反例）**：归一化和 sigmoid 各起一个 kernel、各读一遍拼接输入 → 输入读两次 + 中间结果落 HBM → ✅ 正解即本条：单 kernel 按 HALF_DIM 逻辑切分，两半分别走归一化/sigmoid 流水线，输入只读一次。

**已知实例**：normalization §raw (`experiences/normalization/exp_fused_rmsnorm_sigmoid_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel assumes the input tensor 'QQ' contains data for both operations concatenated along the last dimension. Ensure that the DIM parameter is an even number.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../memory-sram/exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../memory-sram/exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Memory Access Optimization: sub blocking strategy](../memory-sram/exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations
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
- 相关主题: [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, normalization, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, normalization, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, normalization, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, normalization, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations

<!-- okf:related:end -->
