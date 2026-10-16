---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fused split and dual-norm single-pass load"
description: "normalization 场景下，Memory Access Optimization: fused split and dual-norm single-pass load，收益 35%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fused split and dual-norm single-pass load

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16，低秩维度在 8-128 量级；失效边界：q_lora_rank + kv_lora_rank + qk_rope_dim 之和超过硬件单次向量加载宽度时单遍顺序读被打断，需分段加载并重算收益；输入 fused buffer 非连续时顺序突发访问失效）

### 原则
- **kernel fusion**：Fuses tensor splitting, two RMSNorm operations (for inputA and inputB-nope), and a direct copy operation (for inputB-pe) into a single kernel. This reduces global memory reads of the input `fused_ptr` from multiple passes to a single pass, significantly lowering memory bandwidth pressure.
- **memory access pattern**：Implements a strictly sequential access pattern for the input tensor `fused_ptr`. By loading the inputA segment, followed immediately by the inputB-nope segment, and then the inputB-pe segment, the kernel maximizes DRAM burst utilization and cache hit rates.
- **numerical precision**：Explicitly casts input data to `tl.float32` before computing the variance (`q_var`, `k_var`). This prevents precision loss and potential overflow during the accumulation of squared values, which is critical when inputs are `fp16` or `bf16`.
- **compile time branching**：Utilizes `tl.constexpr` for `Q_HAS_BIAS` and `K_HAS_BIAS`. This allows the Triton compiler to generate specialized code paths where bias addition logic is either inlined or completely omitted, avoiding runtime dynamic branching overhead.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/191-fused_split_qk_norm_kernel.py
```

**坏实践（反例）**：切分与两个归一化拆成多个 kernel，fused 输入被反复读取
```python
q      = tl.load(fused_ptr + q_off + offs)   # kernel A: 切分 + 归一化 inputA
k_nope = tl.load(fused_ptr + kn_off + offs)  # kernel B: 切分 + 归一化 inputB-nope
tl.store(kpe_ptr + offs, tl.load(fused_ptr + kp_off + offs))  # kernel C: 拷贝 inputB-pe
```
后果：同一 fused buffer 被多遍 GM 读取，带宽压力随 kernel 数成倍增加，launch 开销叠加。✅ 正解：单 kernel 内按 inputA → inputB-nope → inputB-pe 的顺序一次性扫过 fused_ptr（严格顺序访问打满 DRAM 突发）；方差（q_var、k_var）显式转 fp32 计算防溢出；偏置有无用 `tl.constexpr` 分支在编译期裁剪。

**已知实例**：normalization §raw (`experiences/normalization/exp_fused_split_qk_norm_001.yaml`)
- 声称性能收益：35%
- 备注：对低秩维度在 8-128 量级的推理场景非常有效。需保证 `q_lora_rank`、`kv_lora_rank`、`qk_rope_dim` 之和不超过硬件 Cube/向量单元支持的最大向量加载宽度，以获得最佳性能。

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 normalization
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](../algorithm-numerics/split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [计算优化: phase split 与 barrier 外提](../compute-pipeline/exp_kda_bwd_wy_structure_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_layernorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_reduce_split_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_rmsnorm_backward_npu_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](../compute-pipeline/exp_rmsnorm_bwd_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_kv_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_rmsnorm_rope_001.md) — 主题相关：共享 normalization, reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](../compute-pipeline/exp_sda_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](../compute-pipeline/exp_solve_tril_003.md) — 同属优化知识，主题相关
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 normalization, 访存效率
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
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 normalization, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
