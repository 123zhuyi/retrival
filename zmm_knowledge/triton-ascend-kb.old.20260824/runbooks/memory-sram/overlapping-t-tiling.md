---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 重叠 T-tiling 保证跨 tile 滑动窗口连续性
description: 对序列数据做 T 维度切分时，相邻 tile 保留 W-1 个元素重叠区，避免每个 tile 独立从状态缓存重新加载历史 token。
tags: [overlapping_tiling, t_tile, sliding_window, causal_conv1d, ssm, state_cache]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## CT-14 重叠 T-tiling 保证跨 tile 滑动窗口连续性

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：序列处理 kernel 做了 T 维度 tile 化，滑动窗口/卷积/状态机需要跨 tile 历史；失效边界：无 T 维度切分，或使用寄存器滑动窗口）

<原则：对序列数据做 T 维度切分时，相邻 tile 之间若直接断开，滑动窗口操作在边界处会丢失历史上下文，导致结果错误。相邻 tile 之间保留 W-1 个元素的重叠区，后续 tile 向前多加载 W-1 个 token（等于上一 tile 结尾），既保证连续性，又避免从外部状态缓存重新加载历史。>

**核心思想**：

```
T-tile 0: [history | t0 ..... t255]
T-tile 1:          [t254, t255 | t256 ..... t511]
                   ↑----------↑
                W-1 个重叠元素
```

**正例**：

```python
# T-tile 0: 从状态缓存拼接历史
x_b = tl.insert_slice(
    tl.load(state_ptr, offsets=(0, state_len - (W-1)), block_shape=(D_CHK, (W-1)+T_CHK)),
    tl.load(x_ptr, offsets=(0, 0), block_shape=(D_CHK, T_CHK)),
    (0, W-1), (D_CHK, T_CHK))

# T-tile i (i>0): 向前偏移 W-1，自然包含前 tile 结尾
x_b = tl.load(x_ptr, offsets=(0, ti*T_CHK - (W-1)), block_shape=(D_CHK, T_CHK+W-1))
```

**适用场景**：

- 序列处理 kernel 做了 T 维度 tile 化（Causal Conv1D、SSM chunk scan）
- 滑动窗口 / 卷积 / 状态机需要跨 tile 历史上下文
- W 越小（2-4），额外 buffer 开销越低（~1-3% UB 增量）

**收益**：相比 naive 方案（每个 tile 独立从状态缓存取 W-1 个历史 token），节省 `(num_T_tiles - 1) × (W-1) × D_CHK_SIZE` 次全局内存 load。典型场景节省 15-25% 内存带宽。

**与寄存器滑动窗口的关系**：

- 重叠 T-tiling：用于**有 T-tiling** 的 Update kernel（跨 tile 边界连续性）。
- 寄存器滑动窗口：用于**无 T-tiling** 的 Prefill kernel（逐 token 顺序处理）。

**坏实践（反例）**：
```python
# T-tile 之间无重叠，每个 tile 只加载自己的 T_CHK 个元素
x_b = tl.load(x_ptr + ti * T_CHK * stride, block_shape=(D_CHK, T_CHK))
```
后果：边界处丢失 W-1 历史，结果错误。✅ 正解即 T-tile 间保留 W-1 重叠区。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_causal_conv1d_update_004` — Memory Access Optimization （收益 减少约 (num_T_tiles - 1) × (W-1) × D_CHK_SIZE 次全局内存 load，典型场景节省 15-25% 内存带宽）
  - 重叠 T-tiling 是 Update kernel 在 BDT 布局下实现高效 tile 并行化的核心设计。与 Prefill kernel（一个 program 处理整条序列，无需 T-tiling）形成对比。该模式是序列处理 kernel 做 T 维度切分时的通用范式。

<!-- okf:related:start -->

# 相关

- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 sliding_window
- 实践案例: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 主题相关：共享 sliding_window
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 ssm
- 开发指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 ssm
- 开发指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 ssm
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](../compute-pipeline/scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [Causal Conv1D Compute 优化聚合](../compute-pipeline/traj-conv-compute.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Causal Conv1D Prefill/Update 跨 Kernel 优化迁移](../compute-pipeline/traj-conv-cross_kernel.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](../compute-pipeline/traj-conv-scheduling.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 ssm
- 相关主题: [SSM Compute 优化聚合](../compute-pipeline/traj-ssm-compute.md) — 主题相关：共享 ssm
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](../compute-pipeline/traj-ssm-scalar_elimination.md) — 主题相关：共享 ssm
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 ssm
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](ub-aware-block-adaptive.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
