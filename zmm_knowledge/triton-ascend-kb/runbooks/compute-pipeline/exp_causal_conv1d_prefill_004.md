---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: pointer base precompute'
description: 'conv 场景下，Compute Optimization: pointer base precompute，收益 减少 10-20% 的标量地址计算指令（取决于循环体大小），在大 seq_len 场景收益更显著'
tags: [AICore 单元利用/Scalar 削减, convolution-operations, conv]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: pointer base precompute

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：循环体极小（仅 1-2 次迭代）时预计算开销与收益相抵；地址表达式本身可被编译器常量折叠的场景无额外收益；seq_len < 16 时收益可忽略）

### 原则
- **pointer base precompute**：将循环外可计算的地址分量（batch start offset、D-tile offset）预先计算为基址指针，循环内仅做廉价的 `+ i * stride` 偏移加法。循环外的指针计算（`start * stride + d_offs * stride_dim`）只需执行一次，循环内每次迭代省去整数乘法和完整地址重组。
- **loop invariant hoisting**：属于编译器优化的手动实现——将循环不变量（loop-invariant）显式提升到循环外。在 Triton kernel 中，编译器不一定能自动识别指针表达式中的不变量（尤其当涉及多维 stride 时），手动预计算确保优化必定生效。
- **stride simplification**：循环内的地址增量简化为单一 stride 乘法：`x_ptrs + i * stride_x_tokens`。当 stride_x_tokens 是编译期常量时，`i * stride` 可被强度削减为累加（`addr += stride`），进一步减少乘法指令。
- **applicability general**：适用于所有含 token/row 维度循环的 kernel（attention、normalization、elementwise over seq_len），只要循环体内需要按照固定 stride 访问不同位置的元素。

### 代码/骨架出处
- `precompute_pattern`:
```python
# 循环外：预计算基址（只一次）
x_ptrs = x_ptr + start * stride_x_tokens + d_offs * stride_x_dim
y_ptrs = y_ptr + start * stride_y_tokens + d_offs * stride_y_dim

# 循环内：仅廉价偏移
for i in range(seq_len):
    col3 = tl.load(x_ptrs + i * stride_x_tokens, mask=mask_d)
    # ... compute ...
    tl.store(y_ptrs + i * stride_y_tokens, acc, mask=mask_d)
```

**坏实践（反例）**：循环内每次迭代都写完整地址表达式 `x_ptr + (start+i)*stride_x_tokens + d_offs*stride_x_dim` → 每次迭代重复整数乘加与地址重组，标量单元成为瓶颈（浪费 10-20% 标量指令）→ ✅ 正解即本条：循环外预计算基址指针，循环内仅做 `+ i*stride` 廉价偏移。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_prefill_004.yaml`)
- 声称性能收益：减少 10-20% 的标量地址计算指令（取决于循环体大小），在大 seq_len 场景收益更显著
- 备注：此优化在遗传搜索 Round 2 中被引入，与 DIM_IS_MULTIPLE 共同贡献了 26% 的 Round 1→Round 2 性能提升。是通用模式，不限于 conv1d。

<!-- okf:related:start -->

# 相关

- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block pointer abstraction](../memory-sram/exp_attn_ptr_setup_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
- 相关主题: [Memory Access Optimization: block pointer utilization](../memory-sram/exp_idx_copy_001.md) — 同属优化知识，主题相关
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](../meta-skills/precompute-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
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
