---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: redundant mask elimination by matrix property'
description: 'solve_tril 场景下，Compute Optimization: redundant mask elimination by matrix property，收益 1.33x-1.87x（BT=16 kernel）；~1.40x（BT=64 kernel）；BT=32 无此冗余故无变化'
tags: [AICore 单元利用/Scalar 削减, linear-algebra-operations, solve_tril]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: redundant mask elimination by matrix property

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：仅当零元由矩阵数学结构保证（而非运行时条件）才可删 mask——结构前提破坏即错结果；实测收益与迭代次数正相关（BT=16 省 14 条 where，1.33x-1.87x；BT=32 原本无此冗余）；编译器若已能自动消除则重复操作无额外收益）

### 原则
- **redundant mask elimination by matrix property**：当矩阵具有已知结构零元（严格下三角、对角、上三角等）时，对已为零的元素再次施加的 `tl.where(mask, val, 0.)` 是冗余 ALU 指令。Triton 编译器可能无法在运行时变量 `i` 驱动的循环中优化掉这类掩码。直接删除冗余掩码，依赖矩阵结构本身保证零值位置正确。
- **strictly lower triangular case**：对于严格下三角矩阵 A（A[i,j]=0 for j≥i），循环体内 `b_a = tl.where(o_i < i, b_a, 0.)` 是恒等变换——`b_a` 从 A 加载后取反，j≥i 位置已经是 0。删除后精度不变、性能提升 1.33x-1.87x（BT=16, 14 次迭代节省 14 条 where 指令）。
- **applicability general**：适用于所有处理具备已知结构零元的矩阵 kernel：三角矩阵（tril/triu）、对角矩阵、带状矩阵、mask 矩阵。只要矩阵元素在特定位置的零值由数学保证而非运行时条件决定，对应的保护性 mask 即可删除。
- **tl where cost**：`tl.where(condition, a, b)` 在 Ascend NPU 上需要：(1) condition 比较指令；(2) 逐元素 select 指令。对 16 元素向量每次迭代省 16 次 cmp + 16 次 select，14 次迭代累计 448 条 scalar 指令。

### 代码/骨架出处
- `before`:
```python
# 循环体内：从 A 加载行、取反、再加 where mask
for i in range(2, min(16, T - i_t * 16)):
    b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
    b_a = tl.where(o_i < i, b_a, 0.)     # ← 冗余：A[i,j≥i] 本身就是 0
    b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
    b_A = tl.where((o_i == i)[:, None], b_a, b_A)
```
- `after`:
```python
# 删除冗余 tl.where，依赖矩阵结构保证零值
for i in range(2, min(16, T - i_t * 16)):
    b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
    # b_a[j≥i] 本身就是 0（严格下三角），无需额外 mask
    b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
    b_A = tl.where((o_i == i)[:, None], b_a, b_A)
```

**坏实践（反例）**：即本条主题——对严格下三角矩阵已为零的位置再套 `tl.where(mask, val, 0.)` 保护（从其他 kernel 复用代码时带入的冗余）→ 每次迭代多 16 次 cmp + 16 次 select，14 次迭代累计 448 条标量指令 → ✅ 正解即本条：删掉冗余 mask，靠矩阵结构保证零值正确（删除前必须确认零元由数学结构保证）。

**已知实例**：solve_tril §raw (`experiences/solve_tril/exp_solve_tril_001.yaml`)
- 声称性能收益：1.33x-1.87x（BT=16 kernel）；~1.40x（BT=64 kernel）；BT=32 无此冗余故无变化
- 备注：此为 V1 优化。从 fla 仓库原始代码中识别出的冗余模式——原始代码直接复用了 merge kernel 的代码模式，未针对 16×16 做差异化处理。BT=32 kernel 原本就没有此掩码。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 solve_tril
- 相关主题: [Memory Access Optimization: redundant access elimination](../memory-sram/exp_masked_fill_001.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](../memory-sram/exp_solve_tril_002.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — 同属优化知识，主题相关
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](constexpr-branch-elimination.md) — 同属优化知识，主题相关
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
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减, linear-algebra-operations, solve_tril
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](int-compare-cast-float32.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
