---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '标量消除: i64/i32 标量降级定向消除'
description: 'ssm 场景下，标量消除: i64/i32 标量降级定向消除，收益 233%'
tags: [AICore 单元利用/Scalar 削减, ssm]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 标量消除: i64/i32 标量降级定向消除

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：i64→i32 要求地址范围确实在 i32 内，超大张量需复核；NK/NV 编译期常量要求 K/V 在编译期可知；关键红线是保留 tl.dot 架构——逐元素展开虽降 scalar ratio 但丧失 Cube 加速，性能严重回退）

### 原则
- **背景**：chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2在Ascend上Task Duration从~169k μs经7轮优化降至~50.7k μs（3.33x）
- **核心优化1**：消除i64地址计算。将所有涉及i_tg/bos/eos的偏移保持i32，避免int64算术标量降级
- **核心优化2**：i32整数比较转fp32。所有mask生成（m_t, m_k, BT_mask, BK_mask, m_A等）使用fp32比较
- **核心优化3**：tl.cdiv(inputB,BK)和tl.cdiv(inputC,BV)改为编译期常量NK/NV，避免运行时除法取余
- **核心优化4**：min(T, expr)改为tl.where向量化条件选择
- **核心优化5**：移除b_h/b_dh/b_dv等冗余compile_hint，释放编译器优化空间
- **核心优化6**：使用other=0.0替代load后的冗余tl.where清零
- **推荐顺序**：先消除i64（收益最大）→ i32转fp32 → tl.cdiv/min优化 → 移除hint → autotune

### 代码/骨架出处
- `Triton`:
```python
# i64 → i32
i_tg = i_b * NT + i_t  # 保持 i32

# i32 比较 → fp32 比较
m_t = o_t.to(tl.float32) < T.to(tl.float32)

# tl.cdiv → 编译期常量
NK = K // BK
NV = V // BV
for i_k in range(NK):
    for i_v in range(NV):
        ...

# min → tl.where
t_end = i_t * BT + BT
t_last = tl.where(t_end < T, t_end, T) - 1
```

**坏实践（反例）**：偏移计算混入 i64（i_tg/bos/eos）、mask 用 i32 比较、循环上界用运行时 `tl.cdiv` → NPU 标量单元对 i64/i32 运算降级，实测 7 轮优化前 Task Duration ~169k μs → ✅ 正解即本条：偏移保持 i32、mask 比较转 fp32、cdiv 改编译期常量 NK/NV、min 改 `tl.where`（优化后 ~50.7k μs，3.33x）。

**已知实例**：ssm §raw (`experiences/ssm/exp_kda_bwd_wy_scalar_opt_001.yaml`)
- 声称性能收益：233%
- 备注：此bundle针对wy_dqkg_fused反向kernel。关键原则是保留tl.dot架构（不逐元素展开），定向消除导致标量降级的i64/i32操作。逐元素展开虽能降低scalar ratio，但会丧失Cube加速，性能严重回退。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](exp_perf_012.md) — 主题相关：共享 aicore 单元利用/scalar 削减

<!-- okf:related:end -->
