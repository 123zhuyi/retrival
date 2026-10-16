---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: bdt layout rationale'
description: 'conv 场景下，Scheduling Optimization: bdt layout rationale，收益 BDT 布局 + 3D 任务分解 + 双缓冲状态更新组合贡献约 30-40% 整体性能提升（vs BTD + naive tiling）'
tags: [多核并行与负载均衡, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: bdt layout rationale

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：仅适用于 seq_len ≤ 256 的 update/decode 场景，长序列 prefill 应选 BTD + 2D grid 的互补方案；BDT 布局下 D_CHK_SIZE 必须保持小值，需要通用大 tile 的场景不适用）

### 原则
- **bdt layout rationale**：Update kernel 采用 BDT `[B, D, T]` 布局而非 Prefill 的 BTD `[B, T, D]`。原因是 Decode 阶段 seq_len 小（1 token），但 D 维度大（4096）。BDT 布局下 D 维度连续（stride=1），有利于：(1) 权重 tile `[D_CHK_SIZE, W]` 一次 block load；(2) x_b `[D_CHK_SIZE, buffer_len]` 一次加载；(3) 输出 `[D_CHK_SIZE, T_CHK_SIZE]` 一次 store。BTD 布局对长序列的逐 token 处理更友好（同一 token 的所有 D 维度连续），但对短序列的矩阵式卷积不如 BDT。
- **st store head tile size precomputation**：在 host 端预计算 `ST_STORE_HEAD_TILE_SIZE` 作为编译期常量传入 kernel，避免 kernel 内使用 `tl.maximum` 处理负偏移。公式：`ST_STORE_HEAD_TILE_SIZE = width if (seqlen % T_CHK_SIZE) > width else (width - seqlen % T_CHK_SIZE) % T_CHK_SIZE`。在 Ascend 平台上，避免动态负偏移计算可以显著减少 scalar 指令数。用于处理状态更新时 tile 起始偏移为负数（`t_off < -(W-1)`）的特殊情况。
- **double buffered state update**：kernel 写入独立的 `conv_state_update` buffer 而非直接修改 `conv_state`，由宿主函数通过 `conv_state.copy_(conv_state_update)` 一次性同步。这允许多个 program 并发写入不同区域而无竞争，保证状态更新的原子性。类似双缓冲机制，读写分离消除了 WAR（Write-After-Read）依赖。
- **three dimensional task decomposition**：使用 1D grid + strided loop 将 `batch × NUM_D_CHK × NUM_T_CHK` 三维任务映射到硬件。每个 program 通过 `for task in tl.range(pid, total_tasks, num_programs)` 处理多个 tile。相比 2D grid `(batch, cdiv(dim, BLOCK_D))`：(1) 更灵活地处理 D 和 T 两个维度的 tile 化；(2) D_CHK_SIZE=16（小 tile）搭配 T_CHK_SIZE=256（大 tile）的异构 tile 策略被自然支持；(3) 当 total_tasks < num_programs 时自动处理空闲核。
- **d tiling size rationale**：D_CHK_SIZE=16 而非更大的值（如 128 或 256）的原因：BDT 布局下 T 维度在最内层（stride=1），大的 D_CHK_SIZE 会导致跨步访问（各行间距 = seq_len）。D_CHK_SIZE=16 保证每个 D-tile 内的访问连续性好，同时配合 T_CHK_SIZE=256 让每个 tile 覆盖足够多 token 以减少 tile 数量。

### 代码/骨架出处
- `st_store_head_precompute`:
```python
# host 端预计算，避免 kernel 内动态负偏移
ST_STORE_HEAD_TILE_SIZE = (
    width if (seqlen % T_CHK_SIZE) > width
    else (width - seqlen % T_CHK_SIZE) % T_CHK_SIZE
)
```
- `three_d_task_decomposition`:
```python
# 1D grid → 3D task decomposition
pid = tl.program_id(0)
total_tasks = batch * NUM_D_CHK * NUM_T_CHK
for task_id in tl.range(pid, total_tasks, tl.num_programs(0)):
    bi = task_id // (NUM_D_CHK * NUM_T_CHK)
    di = (task_id // NUM_T_CHK) % NUM_D_CHK
    ti = task_id % NUM_T_CHK
```

**坏实践（反例）**：在 seq_len≤256 的 decode 场景沿用长序列 prefill 的 BTD 布局 + 大 D-tile（如 256），且多个 program 直接写同一状态 buffer → 大 D-tile 造成跨步访问（行间距 = seq_len）、并发写产生 WAR 竞争 → ✅ 正解即本条：BDT 布局 + D_CHK_SIZE=16 小 tile + 独立 update buffer 双缓冲写回。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_007.yaml`)
- 声称性能收益：BDT 布局 + 3D 任务分解 + 双缓冲状态更新：组合贡献约 30-40% 整体性能提升（vs BTD + naive tiling）
- 备注：此经验是关于 Update kernel 的整体架构设计决策，涵盖数据布局、tile 大小、Grid 策略、状态更新机制等维度。这些设计决策相互耦合：BDT 布局决定了 D_CHK_SIZE 必须小（避免跨步），小 D_CHK_SIZE 又决定了需要 T-tiling 来并行化，而 T-tiling 引入了重叠加载和状态更新的复杂性。

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
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡

<!-- okf:related:end -->
