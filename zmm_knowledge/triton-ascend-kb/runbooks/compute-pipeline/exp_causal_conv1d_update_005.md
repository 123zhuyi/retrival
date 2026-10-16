---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: dim is multiple mask elimination'
description: 'conv 场景下，Compute Optimization: dim is multiple mask elimination，收益 减少 5-15% 的标量指令数（取决于 tile 数量），在 D 维度较大且整除 D_CHK_SIZE 时收益更显著'
tags: [AICore 单元利用/Scalar 削减, convolution-operations, conv]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: dim is multiple mask elimination

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：dim 非 D_CHK_SIZE 整数倍时必须保留 mask（constexpr 自动回退）；D 维 tile 数少时收益有限；`boundary_check` 是编译期行为，运行时才能确定的形状无法静态消除）

### 原则
- **dim is multiple mask elimination**：当 `dim % D_CHK_SIZE == 0`（或 `dim % BLOCK_D == 0` for Prefill）时，通过编译期常量 `DIM_IS_MULTIPLE: tl.constexpr` 跳过 D 维度的 mask 检查。在 Update kernel 中 D_CHK_SIZE=16，dim=4096 时能整除，可消除所有 load/store 的 `(offset0 < dim)` mask 运算。这直接减少每条 load/store 指令的标量地址计算和条件判断开销。
- **cross kernel applicability**：此优化来源于 Prefill kernel（`_causal_conv1d_prefill_kernel` 中 BLOCK_D=1024，当 dim%1024==0 时 DIM_IS_MULTIPLE 为 True），可无损迁移到 Update kernel。Update kernel 在加载权重（x_b 的 `offsets=(di*D_CHK_SIZE, ...)` 带 mask）、存储输出（`boundary_check=(0,1)` 带 mask）时都有 D 维度边界检查。
- **compile time specialization**：使用 `tl.constexpr` 声明 DIM_IS_MULTIPLE，编译器在生成 NPU 指令时直接消除 mask 分支，而非在运行时判断。与 `@triton.heuristics` 将 runtime 值转为 constexpr 的模式一致。
- **applicable locations**：Update kernel 中适用位置：(1) 权重加载 `tl.load(make_block_ptr(weight_ptr, ...))` 的 boundary_check；(2) x_b 加载的 D 维度 offsets 计算中的 mask；(3) 输出存储 `tl.store(make_block_ptr(out_ptr, ...))` 的 boundary_check。

### 代码/骨架出处
- `host_side`:
```python
DIM_IS_MULTIPLE = (dim % D_CHK_SIZE == 0)
```
- `kernel_side`:
```python
if not DIM_IS_MULTIPLE:
    mask_d = d_offs < dim
else:
    mask_d = None
w = tl.load(make_block_ptr(..., boundary_check=(0, 1) if not DIM_IS_MULTIPLE else ()))
```

**坏实践（反例）**：dim=4096 已是 D_CHK_SIZE=16 的整数倍，权重加载与输出存储仍始终带 `boundary_check=(0,1)` 和 `(offsets < dim)` mask → 每次块传输附带边界判断，scalar 指令浪费 5-15% → ✅ 正解即本条：`DIM_IS_MULTIPLE` constexpr 下 `boundary_check=()`、`mask=None`。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_005.yaml`)
- 声称性能收益：减少 5-15% 的标量指令数（取决于 tile 数量），在 D 维度较大且整除 D_CHK_SIZE 时收益更显著
- 备注：此优化是通用模式，适用于所有 D/BD 维度 tile 化且维度大小是 tile size 整数倍的 kernel。实现极简：host 端计算 `DIM_IS_MULTIPLE = (dim % D_CHK_SIZE == 0)`，kernel 端用 `if not DIM_IS_MULTIPLE: mask = (offsets < dim)` 条件编译。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — 同属优化知识，主题相关
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](int-compare-cast-float32.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
