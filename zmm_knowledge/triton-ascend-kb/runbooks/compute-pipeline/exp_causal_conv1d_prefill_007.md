---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: dim is multiple mask elision'
description: 'conv 场景下，Compute Optimization: dim is multiple mask elision，收益 减少 5-15% 标量指令（取决于 tile 数量），大 D 维度场景收益更显著'
tags: [AICore 单元利用/Scalar 削减, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: dim is multiple mask elision

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：dim 非 BLOCK_D 整数倍时必须走带 mask 的通用路径（constexpr 自动回退，无收益）；tile 数很少（dim≈BLOCK_D）时 mask 开销占比小、收益 <5%；运行时数据依赖的 mask（如变长序列边界）无法用此法消除）

### 原则
- **dim is multiple mask elision**：当 `dim % BLOCK_D == 0`（如 dim=4096, BLOCK_D=1024）时，通过编译期常量 `DIM_IS_MULTIPLE: tl.constexpr` 跳过所有 load/store 的 D 维度 mask 检查。Triton 编译器在编译期消除 `if DIM_IS_MULTIPLE` 分支，mask 比较和 predicated load/store 指令完全消失。对 dim=4096（最常见 LLM hidden size），此优化始终生效。
- **prefill vs update unified**：此模式在 Prefill kernel（BLOCK_D=1024）和 Update kernel（D_CHK_SIZE=16）中通用。共性：只要 D 维度 tile size 能整除 dim，即可通过 constexpr 消除 mask。分别在 Round 2 遗传搜索中贡献了 26% 的性能提升（与指针预计算组合）。
- **scalar instruction reduction**：免去 mask 后每条 load/store 省去：(1) d_offs < dim 的向量比较指令；(2) predication 的 mask 寄存器操作；(3) 越界元素的 other=0.0 填充逻辑。对于 25-60 行循环体的 kernel，每 token 有 ~10 次带 mask 的 load/store，累计省去大量 scalar 指令。
- **generality note**：适用于所有做 D/BD 维度 tile 化的 kernel（matmul、attention、normalization、elementwise、rope），只要 tile size 能整除维度大小。是 Triton-Ascend kernel 的标准优化模式。

### 代码/骨架出处
- `prefill_version`:
```python
# host 端
DIM_IS_MULTIPLE = (dim % BLOCK_D == 0)  # BLOCK_D=1024, dim=4096 → True

# kernel 端
@triton.jit
def kernel(..., DIM_IS_MULTIPLE: tl.constexpr, BLOCK_D: tl.constexpr):
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    if DIM_IS_MULTIPLE:
        mask_d = None       # 所有 load/store 无需边界检查
    else:
        mask_d = d_offs < dim
    x = tl.load(ptr + d_offs, mask=mask_d)
```

**坏实践（反例）**：dim=4096、BLOCK_D=1024 已整除，仍对所有 load/store 挂 `mask=d_offs<dim` → 每条访存多一次向量比较 + predication 寄存器操作，循环体内累计浪费 5-15% 标量指令 → ✅ 正解即本条：host 端算 `DIM_IS_MULTIPLE` constexpr，整除时 `mask_d=None`。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_prefill_007.yaml`)
- 声称性能收益：减少 5-15% 标量指令（取决于 tile 数量），大 D 维度场景收益更显著
- 备注：此经验是 DIM_IS_MULTIPLE 模式的原始出处（Prefill kernel Round 2）。Update kernel 版本见 exp_causal_conv1d_update_005。两者共用一个技术原理，仅 tile size 不同（1024 vs 16）。

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
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减, conv, convolution-operations
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
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](int-compare-cast-float32.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
