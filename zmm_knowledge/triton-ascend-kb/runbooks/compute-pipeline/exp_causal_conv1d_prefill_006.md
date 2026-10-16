---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: conv bias silu triple fusion'
description: 'conv 场景下，Compute Optimization: conv bias silu triple fusion，收益 消除 2 个额外 kernel launch（延迟降低约 30-40%）；Bias 寄存器驻留节省重复 load'
tags: [AICore 单元利用/Scalar 削减, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: conv bias silu triple fusion

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：强绑定 width=4 的滑窗结构；激活含查表或不可内联的非线性时无法融合；中间结果需被其他 kernel 复用（必须落 HBM）时融合反而阻断流水线拆分）

### 原则
- **conv bias silu triple fusion**：将卷积计算 + Bias 加法 + SiLU 激活融合在单个 kernel 的每条 token 循环内。所有中间结果保留在寄存器中（fp32 精度），只产生一次最终的 fp16/bf16 store。相比 3 个独立 kernel launch（conv → add_bias → silu），消除了 2 次 HBM 写 + 2 次 HBM 读。
- **fp32 activation numerical stability**：SiLU 激活 `acc / (1 + tl.exp(-acc))` 在 fp32 累加器上计算，而非提前 downcast 到 fp16。SiLU 中的 exp 对大负值敏感（exp(-x) → inf when x < -88 in fp16），fp32 提供足够范围避免 overflow。
- **constexpr fusion control**：通过 `HAS_BIAS: tl.constexpr` 和 `ACTIVATION: tl.constexpr` 编译期开关控制融合路径。不需要 bias/activation 的场景编译器完全消除对应代码路径，零额外开销。
- **register resident bias**：bias 向量在卷积循环外一次性加载到寄存器，循环内每次迭代复用。对于 seq_len=2048 的场景，节省了 2047 次重复的 bias load。

### 代码/骨架出处
- `triple_fusion`:
```python
# 循环外：bias 一次性加载，寄存器驻留
if HAS_BIAS:
    bias = tl.load(bias_ptr + d_offs, mask=mask_d)

for i in range(seq_len):
    col3 = tl.load(x_ptrs + i * stride_x, mask=mask_d)
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3  # fp32

    if HAS_BIAS:
        acc = acc + bias                          # fp32 bias add
    if ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))            # fp32 SiLU

    tl.store(y_ptrs + i * stride_y, acc.to(OUT_DTYPE), mask=mask_d)
    col0 = col1; col1 = col2; col2 = col3
```

**坏实践（反例）**：卷积、bias add、SiLU 拆成 3 个独立 kernel → 中间结果多 2 次 HBM 写 + 2 次 HBM 读，且 SiLU 在 fp16 上计算时 `exp(-x)` 对大负值溢出 → ✅ 正解即本条：单 kernel 内 fp32 累加器上完成 卷积+bias+SiLU，bias 循环外一次加载寄存器驻留，constexpr 开关裁剪无用路径。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_prefill_006.yaml`)
- 声称性能收益：消除 2 个额外 kernel launch（延迟降低约 30-40%）；Bias 寄存器驻留节省 2047 次 load
- 备注：此融合模式是 Prefill kernel 的标准融合配置，与 Update kernel 中的 Conv1D+State+SiLU 双缓冲融合互补。两者的共同原则：激活函数必须在 fp32 上计算以保证数值稳定性。

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](../memory-sram/exp_rmsnorm_fused_quant_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减

<!-- okf:related:end -->
