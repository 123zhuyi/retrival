---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '标量消除: block_ptr boundary_check 与 i32 比较定向消除'
description: 'ssm 场景下，标量消除: block_ptr boundary_check 与 i32 比较定向消除，收益 50%'
tags: [AICore 单元利用/Scalar 削减, ssm]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 标量消除: block_ptr boundary_check 与 i32 比较定向消除

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：前提是实测 scalar_ratio 确实高（本例 AIV 41.4%），scalar 占比低的 kernel 上做这套替换收益有限；fp32 比较依赖目标硬件对 fp32 比较的流水支持；转置加载的 mask 布局改写易出错，需重点回归）

### 原则
- **瓶颈**：chunk_kda_bwd_kernel_intra实测AIV scalar_ratio=41.4%，AIC scalar_ratio=31.4%，AIC MAC=2.1%，AIV MTE2=37.9%
- **标量根因**：28+处block_ptr load/store都带boundary_check；10+处i32 LT/GT/LE/GE mask比较；8处min()运行时分支；非SAFE_GATE路径的逐行标量循环
- **核心优化1**：4处store和24+处load从block_ptr改为手动指针+向量化mask，消除boundary_check标量开销
- **核心优化2**：所有mask比较（m_k, m_dA, m_i_diag_qk, m_j_diag_qk等）改为fp32比较
- **核心优化3**：min()全部改为tl.where
- **核心优化4**：tl.cdiv(T_local - i_t*BT, BC)改为基于rows_in_chunk的条件计算
- **推荐顺序**：Phase1 block_ptr store→fp32比较→min→tl.cdiv；Phase2处理MTE2
- **预期效果**：AIV scalar从41.4%降至<20%，AIC scalar从31.4%降至<15%

### 代码/骨架出处
- `Triton`:
```python
# block_ptr store → 手动指针 + mask
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_ti = (i_ti + o_i).to(tl.float32) < T_local.to(tl.float32)
m_k  = o_k.to(tl.float32) < K.to(tl.float32)
BC_BK_mask = m_ti[:, None] & m_k[None, :]
p_dq2 = dq2_local + (i_ti + o_i[:, None]) * H * K + o_k[None, :]
tl.store(p_dq2, b_dq2.to(dq2.dtype.element_ty), mask=BC_BK_mask)

# i32 比较 → fp32
m_i_diag_qk = (o_i_f[:, None] >= o_i_f[None, :]) & ((i_ti + o_i_f[:, None]) < T_local_f)
```

**坏实践（反例）**：28+ 处 block_ptr load/store 全部携带 boundary_check、mask 用 i32 比较、`min()` 运行时分支 → 标量指令占比飙到 41.4%（AIV），Cube/Vector 单元被标量流水饿死 → ✅ 正解即本条：手动指针+向量化 mask 替代 boundary_check、i32 比较转 fp32、min 改 `tl.where`。注意：改写转置 dAqk/dAkk 加载的 mask 布局是本条最易错点。

**已知实例**：ssm §raw (`experiences/ssm/exp_kda_bwd_intra_scalar_opt_001.yaml`)
- 声称性能收益：50%
- 备注：intra kernel与wy_dqkg_fused的差异：intra已经正确使用i32（无i64问题），但标量占比仍极高，说明block_ptr boundary_check和i32比较是主要根因。优化时需特别注意转置dAqk/dAkk加载的mask布局。

<!-- okf:related:start -->

# 相关

- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](../memory-sram/preload-extract-slice-eliminate-reread.md) — 同属优化知识，主题相关
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
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
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, ssm
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](exp_perf_012.md) — 主题相关：共享 aicore 单元利用/scalar 削减

<!-- okf:related:end -->
