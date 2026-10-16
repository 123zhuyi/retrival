---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: Operator Fusion'
description: 'normalization 场景下，Compute Optimization: Operator Fusion，收益 30%'
tags: [AICore 单元利用/Scalar 削减, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: Operator Fusion

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：单 pass 要求整行（n 维）能装入 tile 或分块归约，n 极大时需改两遍/online 方案；`@triton.heuristics` 的特化组合数随可选特征膨胀，编译产物数量需控制；fp32 归约是精度前提）

### 原则
- **Operator Fusion**：Fuses the computation of mean, variance, and the final affine transformation (y = (x - mean) * rstd * w + b) into a single kernel launch. This eliminates the need to write intermediate statistics to global memory and read them back for the normalization step, significantly reducing memory bandwidth pressure.
- **Weight Broadcasting**：Loads the weight (W) and bias (B) tensors once per program instance (per block) and broadcasts them across the blockM dimension. This optimization minimizes redundant memory accesses for the normalization parameters which are constant across the row dimension.
- **Heuristic Specialization**：Utilizes `@triton.heuristics` to generate specialized kernels at compile time for optional features such as Bias (`HAS_BIAS`), Residual connection (`HAS_Z`), and RMS Norm (`IS_RMS_NORM`). This removes dynamic branching overhead from the main computation loop, ensuring the instruction pipeline remains efficient.
- **Grouped Execution**：Implements a grouped execution strategy using `program_id(1)` to handle multiple independent Layer Normalization instances (e.g., in batched or grouped scenarios) within a single grid launch, improving hardware utilization.
- **Precision Management**：Explicitly casts input data to `float32` before computing the sum (mean) and sum of squares (variance). This ensures numerical stability during the reduction operations, which is critical for the convergence of models using lower precision inputs like fp16 or bf16.
- **single pass algorithm**：The kernel implements a one-pass algorithm where the input tile is loaded once to compute both the mean (if applicable) and variance. This significantly reduces global memory access bandwidth compared to a two-pass approach (one pass for statistics, one for normalization), effectively halving the read traffic for the input tensor X.
- **comprehensive operator fusion**：The kernel fuses multiple distinct operations into a single kernel launch LayerNorm/RMSNorm calculation, Affine transformation (Weight/Bias multiplication and addition), and a Gating mechanism (Z * Sigmoid(Z)). This eliminates the need to write intermediate results to high-bandwidth memory (HBM) and reduces kernel launch latency.
- **numerical stability precision**：Intermediate calculations for mean and variance are explicitly upcast to `tl.float32` before the reduction sum (`tl.sum`). This prevents precision loss and overflow/underflow issues that are common when performing reductions directly on fp16 or bf16 data types, ensuring the stability of the backward pass.
- **compile time specialization**：The use of `@triton.heuristics` allows the kernel to specialize at compile time based on the presence of optional arguments (Bias `B`, Residual `Z`, and Norm type). This removes dynamic branching overhead inside the main computation loop, generating highly optimized assembly code for specific model configurations (e.g., SwiGLU vs standard FFN).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/181-_layer_norm_fwd_1pass_kernel_npu.py
```

**坏实践（反例）**：mean/variance 各算一遍（两遍读输入）、统计量写回 HBM 再做归一化、可选特征用运行时 if 分支 → 输入读两遍 + 中间统计量往返 + 分支分歧 → ✅ 正解即本条：单 pass 同块算 mean+variance、仿射/门控融合、`@triton.heuristics` 编译期裁剪可选路径、fp32 归约。

**已知实例**：normalization §raw (`experiences/normalization/exp_layernorm_npu_001.yaml`)
- 声称性能收益：30%
- 备注：This kernel is specifically designed for NPU architectures (indicated by the _npu suffix) and supports flexible execution modes including RMSNorm and pre/post-gating normalization, making it highly suitable for modern Transformer-style architectures.

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
- 相关主题: [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, normalization, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, reduction-operations
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
