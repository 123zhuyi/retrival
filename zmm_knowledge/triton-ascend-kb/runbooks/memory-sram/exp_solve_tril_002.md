---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: preload extract slice eliminate hbm read"
description: "solve_tril 场景下，Memory Access Optimization: preload extract slice eliminate hbm read，收益 1.01x-1.37x（BT=16, V4 vs V1）；1.25"
tags: [访存效率, linear-algebra-operations, solve_tril]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: preload extract slice eliminate hbm read

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，对角块 block_size=16（实测 BT=16/32/64）；失效边界：小 grid（如 B=1、T=512）场景 preload 的寄存器操作开销与省下的 global load 基本抵消（实测仅 1.01x）；该模式依赖 Triton 不支持运行时动态索引这一前提，若编译器行为变化需重估）

### 原则
- **preload extract slice eliminate hbm read**：当矩阵块已在循环外通过单次 `tl.load(block_ptr)` 加载到寄存器后，循环内不应再用 `tl.load(A + row_offset)` 逐行重新从 HBM 读取。虽然 Triton 不支持 `b_A[i, :]` 动态寄存器索引，但可以使用 `tl.extract_slice(neg_A, (i, 0), (1, 16), (1, 1))` + `tl.reshape` 从寄存器块提取行数据，完全消除循环体内的 HBM 访问。
- **why triton cannot dynamic index**：`b_A[i, :]` 在 Triton 中会报错 `ValueError('unsupported tensor index: int32[]')`——i 是运行时 int32，Triton 的 block tensor 不支持运行时动态索引。`tl.extract_slice` 是 Triton 支持的寄存器块子集提取原语，offset 为运行时变量时生成动态偏移指令（开销低于 global load）。
- **preload pattern**：循环前预加载并取反：`neg_A = -b_A`（或保留原值）。循环内 `b_a = tl.extract_slice(neg_A, (i, 0), (1, BLOCK), (1, 1))` + `b_a = tl.reshape(b_a, (BLOCK,))`。BLOCK=16 时 reshape 零开销。
- **multi block extension**：对于分块矩阵（如 32×32 有 2 个对角块、64×64 有 4 个对角块），每个对角块独立预加载并独立 extract_slice，块间无依赖。
- **tradeoff analysis**：小 grid（B=1, T=512）场景下 preload 的额外寄存器操作开销可能与省下的 global load 抵消（实测 1.01x）。大 grid 场景累计效益显著（1.37x）。选择始终使用 extract_slice 可统一代码路径。

### 代码/骨架出处
- `preload_extract`:
```python
# 循环外：预加载整个 16×16 块到寄存器
neg_A = -b_A  # b_A 已通过 tl.load(make_block_ptr(...)) 加载

# 循环内：从寄存器提取，零 HBM 访问
for i in range(2, BT):
    b_a = tl.extract_slice(neg_A, (i, 0), (1, 16), (1, 1))
    b_a = tl.reshape(b_a, (16,))
    b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
    b_A = tl.where((o_i == i)[:, None], b_a, b_A)
```
- `before_hbm_read`:
```python
# 反模式：循环内逐行从 HBM 重新读取（数据已在寄存器中）
for i in range(2, min(16, T - i_t * 16)):
    b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)  # HBM read!
    ...
```

**坏实践（反例）**：循环内逐行从 HBM 重读已预加载到寄存器的块数据（即上方 `before_hbm_read` 反模式）
```python
for i in range(2, BT):
    b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)  # 数据已在寄存器块 b_A 中
```
后果：每行一次多余 GM 读，循环体被内存延迟支配。注意不能写成 `b_A[i, :]`——Triton 对运行时 int32 下标报 `ValueError('unsupported tensor index')`。✅ 正解：循环外单次 `tl.load(block_ptr)` 预加载整块，循环内 `tl.extract_slice` + `reshape` 从寄存器块提取行，循环体零 HBM 访问。

**已知实例**：solve_tril §raw (`experiences/solve_tril/exp_solve_tril_002.yaml`)
- 声称性能收益：1.01x-1.37x（BT=16, V4 vs V1）；1.25x-1.27x（BT=32, V4 vs V1，首次优化）；1.11x-1.13x（BT=64, V4 vs V1）
- 备注：此为 V4 优化的核心。与 V1（exp_solve_tril_001）组合后，ORIGINAL vs V4 累计加速比：BT=16 达 1.42x-2.54x，BT=32 达 1.26x-1.29x，BT=64 达 1.55x-1.57x。

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 solve_tril
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](../compute-pipeline/exp_solve_tril_003.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](../compute-pipeline/exp_solve_tril_004.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — 同属优化知识，主题相关
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
