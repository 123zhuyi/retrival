---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fused residual-add RMSNorm with in-place writeback"
description: "normalization 场景下，Memory Access Optimization: fused residual-add RMSNorm with in-place writeback，收益 35%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fused residual-add RMSNorm with in-place writeback

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093，fp16/bf16，hidden dim N≥1024；失效边界：N<1024 时两遍加载（reduction 一遍、输出一遍）的成本无法摊销，融合收益稀释；单行 N×4B(fp32 累加） 超过 UB 可用容量时需沿 N 切 tile，原地写回模式不变但需重算 UB 预算）

### 原则
- **kernel fusion**：Fuses the residual addition (X+R), variance calculation (sum of squares), reciprocal root mean square computation, and final scaling/weight multiplication into a single kernel. This significantly reduces global memory access overhead by keeping intermediate data in high-speed memory (SRAM/Registers) and avoiding multiple kernel launch latencies.
- **residual inplace update**：Optimizes the data path by writing back the intermediate sum (X+R) to the residual pointer R and the normalized output to X. This eliminates the need for a separate add kernel or extra memory allocation for the residual connection, effectively halving the memory traffic required for the residual path.
- **precision management**：Utilizes float32 precision specifically for the variance accumulation (_var_base) to maintain numerical stability during the reduction of squared values, while performing the final arithmetic and storage in the lower precision input type (e.g., fp16/bf16) to maximize memory bandwidth utilization.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/115-fused_add_rms_norm_kernel.py
```

**坏实践（反例）**：残差相加与归一化拆成两个 kernel，中间结果 X+R 落 GM
```python
# kernel 1: 只算残差和并写回 GM
tl.store(R_ptr + off, x + r)
# kernel 2: 重新从 GM 读 X+R，再算方差与归一化
s = tl.load(R_ptr + off); var = tl.sum(s * s) / N
```
后果：X+R 在 GM 写一次、读一次，残差路径访存流量翻倍，叠加两次 kernel launch 开销。✅ 正解：融合为单 kernel，X+R 保留在 UB/寄存器，同时写回 R 指针（原地更新残差）与 X 指针（归一化输出）；方差累加用 fp32 保数值稳定，最终算术与存储回到 fp16/bf16 打满带宽。

**已知实例**：normalization §raw (`experiences/normalization/exp_fused_rms_norm_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is optimized for scenarios where the hidden dimension N is large enough to amortize the cost of the two-pass loading (once for reduction, once for output). 尤其适用于 Pre-Norm / Post-Norm 结构的归一化场景。

<!-- okf:related:start -->

# 相关

- 实践案例: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](../../ops/add-broadcast-alpha.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 同属优化知识，主题相关
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 同属优化知识，主题相关
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 同属优化知识，主题相关
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 normalization
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 同属优化知识，主题相关
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 normalization
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: chunked scan strategy](../compute-pipeline/exp_chunk_cumsum_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_layernorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: memory alignment](../compute-pipeline/exp_moe_wgrad_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_reduce_split_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](../compute-pipeline/exp_rmsnorm_bwd_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_kv_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](../compute-pipeline/exp_sda_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
