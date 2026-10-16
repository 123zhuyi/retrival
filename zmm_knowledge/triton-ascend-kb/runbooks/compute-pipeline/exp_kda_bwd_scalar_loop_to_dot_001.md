---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '计算优化: 对角块标量循环改 tl.dot'
description: 'ssm 场景下，计算优化: 对角块标量循环改 tl.dot，收益 40%'
tags: [AICore 单元利用/Scalar 削减, ssm]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 计算优化: 对角块标量循环改 tl.dot

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：仅适用于对角块计算且存在逐行标量循环的反向 kernel（BC=16 级别）；改为矩阵操作必须保持 gate 归一化点的数值一致性（SAFE_GATE 用 min(BC//2,...) 而非第 0 行），否则精度漂移）

### 原则
- **问题根因**：当SAFE_GATE=False时，对角块计算使用for j in range(BC)逐行标量循环，每次4个load+标量×向量broadcast乘法，不能利用Cube加速
- **核心方案**：将逐行循环改为矩阵操作，一次性加载整个BC×BC对角块，用2次tl.dot替代16次标量循环
- **精度注意**：SAFE_GATE使用min(BC//2, ...)作为归一化点保证数值稳定性，非SAFE_GATE原用第0行。改为矩阵操作需保持数值一致性
- **典型收益**：16次标量循环→2次tl.dot，AIC MAC ratio从2.1%显著提升，AIV scalar大幅下降
- **适用条件**：对角块计算且存在逐行标量循环的SSM/Attention反向kernel

### 代码/骨架出处
- `Triton`:
```python
# 反模式：逐行标量循环
for j in range(0, min(BC, T_local - i_t*BT - i_i*BC)):
    b_dAqk = tl.load(dAqk_local + o_dA + j, mask=m_dA, other=0)
    b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
    m_i = o_i[:, None] >= j
    b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
    p_kj += H*K

# 正模式：矩阵操作
p_dAqk_diag = dAqk_local + i_ti * H * BT + i_i * BC + o_i[:, None] * H * BT + o_i[None, :]
dA_diag_mask = m_dA[:, None] & m_dA[None, :] & (o_i[:, None] >= o_i[None, :])
b_dAqk_diag = tl.load(p_dAqk_diag, mask=dA_diag_mask, other=0.0)
b_k_exp = b_k * b_exp_neg_g
b_dq2 += tl.dot(b_dAqk_diag, b_k_exp) * b_exp_g
```

**坏实践（反例）**：SAFE_GATE=False 路径用 `for j in range(BC)` 逐行标量循环（每行 4 次 load + 标量×向量 broadcast）→ 16 次循环完全绕开 Cube，AIC MAC 占比仅 2.1% → ✅ 正解即本条：整 BC×BC 对角块一次加载，2 次 `tl.dot` 替代 16 次标量循环。注意：gate 归一化点改动须与 SAFE_GATE 路径对比精度。

**已知实例**：ssm §raw (`experiences/ssm/exp_kda_bwd_scalar_loop_to_dot_001.yaml`)
- 声称性能收益：40%
- 备注：此优化将标量×向量broadcast改为矩阵乘，是提升Cube利用率的关键。但需注意gate归一化点的数值稳定性，建议与SAFE_GATE路径对比精度。

<!-- okf:related:start -->

# 相关

- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 同属优化知识，主题相关
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
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 主题相关：共享 ssm

<!-- okf:related:end -->
