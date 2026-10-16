---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: deep operator fusion'
description: 'misc 场景下，Scheduling Optimization: deep operator fusion，收益 35%'
tags: [多核并行与负载均衡, fused-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: deep operator fusion

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：依赖 cube/vector 异步派发的硬件后端；四级融合链中任一 stage 寄存器/UB 超预算则整链失效；小 shape 下计算量小，融合收益退化为仅省 launch 开销）

### 原则
- **deep operator fusion**：Fuses the sequence of Matrix Multiplication, Sigmoid activation, RMSNorm, and Rotary Positional Embedding (RoPE) into a single kernel launch. This eliminates the need to write intermediate results back to High Bandwidth Memory (HBM) after each operation, significantly reducing memory bandwidth pressure and latency.
- **asynchronous pipeline scheduling**：Utilizes `dl.async_task` to dispatch workloads to distinct hardware scopes (`cube` for matrix multiplication and `vector` for element-wise/reduction operations). This allows the NPU to pipeline instructions, enabling the Vector unit to process normalization and RoPE while the Cube unit is handling the heavy GEMM computation, thereby improving overall hardware utilization.
- **hardware specific resource isolation**：Explicitly maps compute-intensive tasks to the Cube engine and logic-intensive tasks (Sigmoid, Norm, RoPE) to the Vector engine. This isolation prevents resource contention and leverages the specific strengths of each processing unit within the Ascend architecture.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/080-fused_sigmoid_single_norm_rope_matmul_kernel.py
```

**坏实践（反例）**：matmul → sigmoid → 归一化 → 位置编码四个 kernel 串行 → 三次中间结果 HBM 往返，且 Cube/Vector 全程串行等待 → ✅ 正解即本条：深度融合 + 计算密集派 cube、逻辑密集派 vector 的资源隔离。

**已知实例**：misc §raw (`experiences/misc/exp_fused_transformer_block_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is highly optimized for Transformer-style architectures where these operations occur sequentially in the attention or MLP blocks. It requires a hardware backend that supports asynchronous task dispatching between Cube and Vector units.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 fused-operations, misc
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: sub blocking strategy](../memory-sram/exp_gn_wb_backward_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: fused lse reduction](../memory-sram/exp_grpo_bwd_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 fused-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 misc, 多核并行与负载均衡
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 misc

<!-- okf:related:end -->
