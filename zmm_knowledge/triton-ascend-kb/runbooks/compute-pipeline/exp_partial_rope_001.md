---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized memory access'
description: 'rope 场景下，Compute Optimization: vectorized memory access，收益 25%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized memory access

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：partial 旋转要求明确的 NOPE_DIM 划分，全旋转场景无跳过空间；NUM_CORES 需匹配物理核数（备注：910B 为 32），错配则负载不均；ROPE_DIM < 32 时向量块配置需重调）

### 原则
- **vectorized memory access**：The kernel utilizes `tl.arange(0, half_dim)` to load and store contiguous chunks of the head dimension (`x_l`, `x_h`, `o_l`, `o_h`) as vectors. This approach maximizes memory bandwidth utilization and enables the hardware to leverage SIMD (Single Instruction, Multiple Data) capabilities for data movement.
- **partial rope efficiency**：The implementation explicitly calculates offsets starting from `NOPE_DIM` (Non-rotated dimensions). By skipping the non-rotated portion of the head dimension during the load/compute/store phases, the kernel optimizes for modern architectures that utilize partial RoPE, saving significant compute cycles compared to a full-dimension rotation.
- **fused arithmetic pipeline**：The rotation calculations ($x \cos - y \sin$ and $y \cos + x \sin$) are performed immediately on the loaded vector registers before storing. This fusion keeps the data in the high-speed register file (or L1 cache) throughout the transformation, eliminating the latency overhead of writing intermediate results back to global memory.
- **vectorized load splitting**：The COS and SIN embeddings are loaded in two separate vectorized operations for the lower (`_l`) and upper (`_h`) halves of the rotary dimension. This aligns memory access patterns with the complex number rotation logic (treating pairs as real/imaginary components), maximizing vector unit utilization and reducing instruction count.
- **explicit core scheduling**：The kernel utilizes a grid-stride loop pattern `range(pid, NUM_BLOCKS, NUM_CORES)` to statically distribute work across the available AI cores. This explicit scheduling minimizes synchronization overhead and ensures optimal load balancing on NPU architectures like the 910 series.
- **block masking efficiency**：Boundary conditions for sequence length are handled via `pos_mask` and `seq_mask` applied during the load of COS/SIN and implicitly passed to the compute function. This avoids branching within the inner loop and maintains high throughput for sequences not perfectly divisible by the block size.

### 代码/骨架出处
- `TritonPath`:
```python
partial_rope_qk_kernel
```

**坏实践（反例）**：对整个 head_dim 做旋转（含不需要旋转的 NOPE 部分）→ 无效计算与访存随 NOPE_DIM 占比线性浪费 → ✅ 正解即本条：从 NOPE_DIM 起算偏移只处理旋转段 + cos/sin 分半向量加载 + grid-stride 核间分摊。

**已知实例**：rope §raw (`experiences/rope/exp_partial_rope_001.yaml`)
- 声称性能收益：25%
- 备注：The `NUM_CORES` parameter is critical for performance and should be tuned to match the physical core count of the specific NPU chip (e.g., 32 for 910B) to maximize parallelism.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope

<!-- okf:related:end -->
