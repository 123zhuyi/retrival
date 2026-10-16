---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: w loop unrolling with threshold'
description: 'conv 场景下，Compute Optimization: w loop unrolling with threshold，收益 D≥2048: 2.3x-3.1x; D≤256: 0.41x-0.75x（需阈值控制）'
tags: [AICore 单元利用/Scalar 削减, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: w loop unrolling with threshold

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：D<1024 时负收益（实测 D=128 仅 0.41x：独立 load 退化为标量加载 + 寄存器压力约 3 倍）；width≠4 时展开表达式需重新生成；必须配 host 端阈值门控，无门控盲展开会在小 D 场景严重回退）

### 原则
- **w loop unrolling with threshold**：当 W=4 且 D>=1024 时，将 `for owi in range(width)` 循环展开为 4 路独立计算表达式 `x_s0*w0 + x_s1*w1 + x_s2*w2 + x_s3*w3`，消除 3 次 tl.trans 和 8 次 extract_slice 的数据编排开销。实测 D=8192 时加速比 3.1x，D=2048 时 2.3x。D<1024 时回退到原始 for 循环路径，避免小 D 场景下独立 tl.load 的启动开销占比过大导致负收益（D=128 时仅 0.41x）。
- **d threshold gating**：在 host 端通过编译期常量 `W_IS_4=(width==4 and dim>=1024)` 控制是否启用展开路径。D=1024 作为阈值有足够安全边际（正收益 case 均在阈值之上，负收益 case 均在阈值之下），改动仅一行，零风险。
- **extract slice elimination**：展开后消除原始路径中每次 W 迭代的 2 次 extract_slice（x_b 切片 + w 切片），W=4 时共消除 8 次 extract_slice。extract_slice 在 Triton IR 层面创建新的 tensor view（修改 offset/stride/shape 描述符），虽不移动数据但编译器必须生成独立的地址计算逻辑，消除后可减少 scalar 单元压力。
- **transpose elimination**：展开后消除 3 次 tl.trans 调用（x_b 转置 [D_CHK, buf]→[buf, D_CHK]、w 转置 [D_CHK, W]→[W, D_CHK]、out 回转），直接将 x_b 保持 [D_CHK_SIZE, buffer_len] 布局进行切片乘加，减少数据路径的中间步骤。
- **fma friendly expression**：展开后的单语句 4 路归约 `x_s0*w0 + x_s1*w1 + x_s2*w2 + x_s3*w3` 比 4 次独立的 `out += x * w` 更容易被编译器优化为并行 FMA 指令。4 个乘法相互独立可并行发射，加法树结构可直接映射到硬件 FMA 单元。

### 代码/骨架出处
- `host_guard`:
```python
# host 端阈值控制（方案 A，推荐）
W_IS_4=(width == 4 and dim >= 1024),
```
- `kernel_branch`:
```python
if W_IS_4:
    x_s0 = tl.extract_slice(x_b, (0, 0), (D_CHK, T_CHK), (1, 1))
    x_s1 = tl.extract_slice(x_b, (0, 1), (D_CHK, T_CHK), (1, 1))
    x_s2 = tl.extract_slice(x_b, (0, 2), (D_CHK, T_CHK), (1, 1))
    x_s3 = tl.extract_slice(x_b, (0, 3), (D_CHK, T_CHK), (1, 1))
    out_block = (x_s0 * w0[:, None] + x_s1 * w1[:, None] +
                 x_s2 * w2[:, None] + x_s3 * w3[:, None])
else:
    for owi in tl.range(0, width):
        ...
```

**坏实践（反例）**：无阈值地对所有 shape 启用 W=4 展开路径 → 小 D（≤256）场景 4 次独立 `tl.load` 替代 `make_block_ptr` 块传输、寄存器中间量增约 3 倍，实测 0.41x-0.75x 负收益 → ✅ 正解即本条：host 端 `W_IS_4=(width==4 and dim>=1024)` 门控，大 D 走展开、小 D 回退原 for 循环。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_002.yaml`)
- 声称性能收益：D≥2048: 2.3x-3.1x; D≤256: 0.41x-0.75x（需阈值控制）
- 备注：此经验为 W=4 场景下的实测优化，含完整的正/负收益分析和根因诊断。核心启示：循环展开的收益取决于 D 维度大小，需要阈值控制来保留正收益、消除负收益。方案 A（D≥1024 阈值）已可落为一行改动。

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
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
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减

<!-- okf:related:end -->
