---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: constexpr dual path full vs boundary'
description: 'solve_tril 场景下，Compute Optimization: constexpr dual path full vs boundary，收益 此模式保证快速路径（99% chunk）达到最优展开性能，避免统一运行时循环导致的性能退化'
tags: [AICore 单元利用/Scalar 削减, linear-algebra-operations, solve_tril]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: constexpr dual path full vs boundary

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：要求 tile 大小是 constexpr 且存在完整/边界两类 chunk；统一数据来源（extract_slice）是前提——若边界路径用 tl.load 反而更贵；BT 很小（如 16）时循环展开收益有限，双路径的代码复杂度需权衡）

### 原则
- **constexpr dual path full vs boundary**：当 kernel 处理固定大小 tile 时，绝大多数 chunk 是完整 tile（99%），仅最后一个可能是边界不完整 tile。用 `if T - chunk_offset >= BT` 分支：(1) 快速路径使用 `range(2, BT)` 其中 BT 是 `tl.constexpr`，编译器完全展开循环 + extract_slice offset 静态优化为零开销；(2) 慢速路径使用运行时 `range(2, min(16, remaining))`，仍从寄存器提取数据。两个路径统一数据来源（均 extract_slice），区别仅在于循环展开程度。
- **constexpr static unroll benefit**：`range(2, BT)` 当 BT 是 constexpr 时，Triton 编译器在编译期：(1) 完全展开循环体；(2) extract_slice 的 offset 参数 `(i, 0)` 中的 i 为编译期常量，生成静态偏移指令（零 ALU）；(3) 消除循环分支预测开销。
- **boundary path still register**：慢速（边界）路径虽然循环未展开，但同样使用 extract_slice 从预加载的寄存器块读取——消除 HBM 访问的收益对所有 chunk 均等。
- **excluded approaches**：被排除的方案：(a) 统一运行时循环（失去 constexpr 展开优化，性能回退到 V1 水平）；(b) 慢速路径保留 tl.load（不如 extract_slice 统一简洁，且边界块 HBM 读取无性能优势）。
- **applicability**：适用于所有以固定 tile 处理数据、且 tile 大小是 constexpr 的 kernel。只要存在'完整 tile'和'边界不完整 tile'两种情况，即可用此双路径模式。

### 代码/骨架出处
- `dual_path`:
```python
neg_A = -b_A  # 预加载到寄存器

if T - i_t * 16 >= BT:
    # 快速路径：完整 chunk，BT 是 constexpr → 循环完全展开
    for i in range(2, BT):
        b_a = tl.extract_slice(neg_A, (i, 0), (1, 16), (1, 1))
        b_a = tl.reshape(b_a, (16,))
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
else:
    # 慢速路径：边界 chunk，运行时循环 → 仍从寄存器读
    for i in range(2, min(16, T - i_t * 16)):
        b_a = tl.extract_slice(neg_A, (i, 0), (1, 16), (1, 1))
        b_a = tl.reshape(b_a, (16,))
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
```

**坏实践（反例）**：为代码简洁用统一运行时循环 `range(2, min(16, remaining))` 处理所有 chunk（V3 方案）→ 失去 constexpr 完全展开与静态偏移，性能回退到未优化水平 → ✅ 正解即本条：`if T - chunk_offset >= BT` 双路径，99% 完整块走 constexpr 展开快速路径，边界块走运行时循环但仍从寄存器 extract_slice 读。

**已知实例**：solve_tril §raw (`experiences/solve_tril/exp_solve_tril_003.yaml`)
- 声称性能收益：此模式保证快速路径（99% chunk）达到最优展开性能，避免统一运行时循环导致的性能退化
- 备注：此为 V4 最终方案的路径选择策略（V4 混合模式）。V3（统一运行时循环）被排除——虽然代码更简洁，但失去 constexpr 展开后性能回退到 V1 水平。

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 solve_tril
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](../memory-sram/exp_fused_split_qk_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](../memory-sram/exp_solve_tril_002.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 同属优化知识，主题相关
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 同属优化知识，主题相关
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 同属优化知识，主题相关
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
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, linear-algebra-operations, solve_tril
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 linear-algebra-operations, solve_tril

<!-- okf:related:end -->
