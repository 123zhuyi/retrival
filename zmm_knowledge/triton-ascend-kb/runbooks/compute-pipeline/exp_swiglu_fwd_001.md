---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: mixed precision computation'
description: 'activation 场景下，Scheduling Optimization: mixed precision computation，收益 15%'
tags: [多核并行与负载均衡, activation-operations, activation]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: mixed precision computation

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：fp32 中间计算是精度前提（备注自述），纯低精度路径 sigmoid 下溢；n_cols < 128 时块配置需重调；BLOCK_SIZE_M 1-32 适配元素级前向）

### 原则
- **mixed precision computation**：The kernel explicitly casts the input tensor 'a' to float32 before applying the silu (sigmoid linear unit) function. This ensures numerical stability for the sigmoid component, which is prone to overflow/underflow in fp16/bf16, while maintaining performance by casting back to the original dtype for the final multiplication.
- **vectorized memory access**：By utilizing a compile-time constant blockSize and tl.arange, the kernel loads and stores data in vectorized chunks. This maximizes memory bandwidth utilization and reduces the number of memory instructions issued compared to scalar operations.
- **operator fusion**：The kernel fuses the silu activation and the element-wise multiplication (gate mechanism) into a single kernel launch. This eliminates the need to write intermediate results to global memory, significantly reducing memory bandwidth pressure and latency.
- **grid stride scheduling**：The kernel implements a grid-stride loop over the row dimension (`for row_task_id in range(pid, num_row_tasks, grid_size)`). This allows a fixed number of thread blocks to process an arbitrary number of rows efficiently, ensuring high utilization even when the total number of row blocks exceeds the grid size.
- **autotuning block m**：The `BLOCK_SIZE_M` is autotuned across a range of values (1 to 32). This is critical for adapting to varying row dimensions (e.g., different batch sizes or sequence lengths in LLMs), optimizing the trade-off between instruction-level parallelism and GPU occupancy.
- **mixed precision computation**：The input tensor `a` is cast to `float32` before the `silu` activation and then cast back to the original dtype. This ensures numerical stability for the sigmoid calculation, which is prone to underflow in fp16/bf16, without sacrificing the memory bandwidth benefits of lower precision storage.
- **vectorized memory access**：The inner loop processes columns in chunks of `BLOCK_SIZE_N`, ensuring that memory loads and stores are coalesced. This maximizes memory bandwidth utilization by accessing contiguous memory locations.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/211-_swiglu_fwd_kernel.py
```

**坏实践（反例）**：激活与门控乘法拆两个 kernel，且 sigmoid 在 fp16/bf16 下计算 → 中间结果落 HBM + 下溢风险 → ✅ 正解即本条：融合 silu+门控 + 升 fp32 算 sigmoid 再降回 + grid-stride。

**已知实例**：activation §raw (`experiences/activation/exp_swiglu_fwd_001.yaml`)
- 声称性能收益：15%
- 备注：This kernel is specifically optimized for the SwiGLU activation function found in modern LLMs. The float32 intermediate calculation is essential for maintaining accuracy during the activation phase.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 activation
- 相关主题: [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](../algorithm-numerics/fp32-precision-contract.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Memory Access Optimization: precision control](../memory-sram/exp_skip_layernorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation, activation-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation, activation-operations
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation

<!-- okf:related:end -->
