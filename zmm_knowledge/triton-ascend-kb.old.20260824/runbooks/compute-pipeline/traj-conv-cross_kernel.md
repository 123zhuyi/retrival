---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: conv
dimension: "2.4"
title: Causal Conv1D Prefill/Update 跨 Kernel 优化迁移
description: 针对 Causal Conv1D 的 Prefill 与 Update 两个 kernel，分析哪些优化可以跨 kernel 迁移（bias 融合、指针预计算）、哪些不能（W 硬编码、2D grid），以及如何根据 seq_len 选择循环维度。
tags: [conv, cross_kernel, causal_conv1d, prefill, update, bias_fusion, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/conv/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Causal Conv1D Prefill/Update 跨 Kernel 优化迁移

## 概述

Causal Conv1D 通常有两个 kernel：处理长序列的 **Prefill** 和处理增量 token 的 **Update**。两者数据布局、循环方向和并行粒度不同，但很多底层优化可以互相迁移。该经验对比了两者的实现差异，指出哪些优化方向具备通用性、哪些只适用于特定 kernel。

核心结论：

- **循环维度选择是根本性架构决策**：Prefill 适合在 Token 维度循环（BTD 布局，连续加载 1024 维），Update 适合在 W（卷积宽度）维度循环（BDT 布局，适合短序列 decode）；
- **bias 融合最容易迁移**：Update kernel 只需在最终 `out_block` 上加 `bias[None, :]`，即可消除一次独立的 bias add kernel；
- **指针基址预计算可减少标量指令**：从 Prefill 的 `x_ptr + start*stride + d_offs*stride_dim` 模式迁移到 Update 的 W 循环内部；
- **seq_len=1 可特殊化为滑动窗口寄存器复用**，但改动较大、优先级低；
- **W 硬编码为 4、2D grid 并行等 Prefill 优化不建议迁移到 Update**。

收益：bias 融合消除 1 次 kernel launch（延迟降低 10–20%）；指针预计算减少 5–10% 标量指令；seq_len=1 特化预计提升 15–25%（低优先级）。

## 共同模式

### 1. Token 循环 vs W 循环：按 seq_len 与 D 选择

- **Prefill（长序列）**：Token 维度循环，每次处理 `BLOCK_D=1024`。BTD 布局下可一次加载 1024 个连续元素（32 条 cache line），无数据编排开销，FMA 映射为单语句 4 路归约。
- **Update（短序列/decode）**：W 维度循环，每次处理 `D_CHK_SIZE=16`。BDT 布局更适合单 token 的内存访问模式，但涉及 11 次 `extract_slice`/`transpose` 操作。

当 `seq_len=1` 时，BDT+W 循环仍是更自然的选择；若强行套用 Prefill 的 Token 循环，会因加载整个 `[D_CHK, 1+W-1]` buffer 而浪费带宽。

### 2. Bias 融合：从 Prefill 迁移到 Update

Prefill kernel 已实现 `acc = acc + bias`（bias 预加载为寄存器）。Update kernel 虽然声明了 `HAS_BIAS: tl.constexpr`，但相关逻辑尚未实现。迁移只需在 W 循环或展开表达式得到 `out_block` 后加一行：

```python
if HAS_BIAS:
    bias = tl.load(bias_ptr + d_offs, mask=mask_d, other=0.0)
    out_block = out_block + bias[None, :]
```

即可消除一次独立的 bias add kernel launch。

### 3. 指针基址预计算减少标量指令

Prefill 中常用的地址计算模式是 `x_ptrs = x_ptr + start*stride + d_offs*stride_dim`。在 Update 的 `for owi in range(width)` 循环中，x_b 切片和 w 切片的基址可以提前计算，避免每次 `extract_slice` 重复做偏移运算。

### 4. seq_len=1 滑动窗口特化（低优先级）

单 token decode 时，矩阵式卷积可退化为 Prefill 风格的滑动窗口寄存器复用：预加载 `col0/col1/col2/col3` 和 `w0/w1/w2/w3`，计算 `out = col0*w0 + col1*w1 + col2*w2 + col3*w3`，然后移位更新状态。由于 Update 通常按 `T_CHK_SIZE=256` 组织，为单个 token 做特化改动较大，优先级较低。

### 5. 不建议迁移的 Prefill 优化

- **W 硬编码为 4**：Update 需要通用 W 支持；
- **2D grid 并行**：Update 场景 `seq_len` 短，T-tile 切分必要，1D strided grid 配合 3D 任务分解更适合。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_causal_conv1d_update_006` | Bias fusion: 消除 1 次 kernel launch（延迟 ↓10–20%）；Pointer precompute: 标量指令 ↓5–10%；Token-loop for seq_len=1: 预计 ↑15–25%（低优先级） | Causal Conv1D Update 与 Prefill 之间的跨 kernel 优化迁移分析，指出可迁移、不可迁移及条件性迁移的优化。 |

## 代表 Kernel

`exp_causal_conv1d_update_006` 没有独立的 extracted kernel 文件，以下为 YAML 中的关键代码片段：

```python
# Update kernel 中实现 Bias 融合（从 Prefill 迁移）
if HAS_BIAS:
    bias = tl.load(bias_ptr + d_offs, mask=mask_d, other=0.0)
    out_block = out_block + bias[None, :]
```

```python
# seq_len=1 时的滑动窗口寄存器复用（概念迁移）
col0, col1, col2, col3 = load_state_registers()
w0, w1, w2, w3 = load_weight_registers()
new_token = tl.load(x_ptr)
out = col0*w0 + col1*w1 + col2*w2 + col3*w3
col0, col1, col2 = col1, col2, col3
col3 = new_token
store_state(col0, col1, col2)
```

## 已知实例

- `exp_causal_conv1d_update_006` — Cross-kernel Optimization Transfer
  - 适用于 Causal Conv1D 的 Prefill/Update 双 kernel 场景，关键参数 `width=4`，任意 `dim` 与 `seqlen`。
  - 来源分析: `causal_conv1d_prefill_GA/conv1d_update_analysis.md` §4-5、§9、§10。

## 关联知识

- 开发指南: [Conv 算子优化](../../reference/op-class-convolution.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 同属优化知识，主题相关
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 prefill
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 prefill
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 causal_conv1d
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv

<!-- okf:related:end -->
