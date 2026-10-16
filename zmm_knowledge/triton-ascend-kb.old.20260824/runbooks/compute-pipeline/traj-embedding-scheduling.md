---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: embedding
dimension: "2.3"
title: Embedding Backward Scheduling 优化聚合
description: 聚合 1 条 embedding backward 在 Ascend 910B 上的 scheduling 经验：2D block tiling + 软件流水 + batched atomic accumulation，处理大 batch 和大 embedding dim 场景。
tags: [embedding, scheduling, backward, atomic, batched, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/embedding/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Embedding Backward Scheduling 优化聚合

## 概述

Embedding backward 是推荐系统和大语言模型中常见的稀疏更新算子：根据 indices 把梯度累加到 embedding 矩阵的对应行。在 Ascend 910B 上，它的核心挑战是**索引可能重复**，导致多个 program 同时写同一个 embedding 行，引发原子竞争。本卡只有 1 条 experience，但覆盖了一个完整的调度优化方案：2D block tiling、软件流水和 batched atomic accumulation。

## 共同模式

### 1. 2D Block Tiling

把 batch 维度和 embedding 维度同时分块：
- batch 方向分块让多个 program 并行处理不同样本；
- embedding 方向分块让每个 program 只负责 embedding 向量的一段，降低 atomic 冲突粒度。

### 2. Batched Atomic Accumulation

不是每个元素单独 atomic add，而是把同一个目标行的一段连续梯度缓存到寄存器/SRAM，然后一次性 batched atomic add。这样可以：
- 减少 atomic 操作次数；
- 提高每个 atomic 事务的利用率；
- 降低内存总线竞争。

### 3. 软件流水隐藏 Atomic 延迟

atomic 操作有较高延迟。通过 `num_stages` 或手动 prefetch，可以在等待当前 atomic 完成的同时准备下一批数据，提升流水线效率。

### 4. 重复索引的影响

当 batch 中重复索引比例高时，atomic 竞争会显著增加。优化方向：
- 预排序 indices 并按目标行分组（类似 MoE binning）；
- 但 embedding backward 通常 indices 较随机，所以 2D tiling + batched atomic 是更通用的方案。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_embedding_backward_001` | 35% | Embedding backward：2D block tiling、软件流水、batched atomic accumulation |

## 代表 Kernel

以下为 `exp_embedding_backward_001` 对应的 kernel：

```python
@triton.jit
def embedding_backward_kernel(
    grad_output_ptr, indices_ptr, grad_weight_ptr,
    num_tokens, embedding_dim,
    stride_go_tok, stride_go_dim,
    stride_gw_row, stride_gw_dim,
    BLOCK_TOK: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    tok_start = pid_t * BLOCK_TOK
    dim_start = pid_d * BLOCK_DIM

    tok_offs = tok_start + tl.arange(0, BLOCK_TOK)
    dim_offs = dim_start + tl.arange(0, BLOCK_DIM)
    tok_mask = tok_offs < num_tokens

    # 加载 indices
    idx = tl.load(indices_ptr + tok_offs, mask=tok_mask, other=0)

    # 加载 grad_output 的 (BLOCK_TOK, BLOCK_DIM) tile
    go = tl.load(grad_output_ptr + tok_offs[:, None] * stride_go_tok +
                 dim_offs[None, :] * stride_go_dim,
                 mask=tok_mask[:, None] & (dim_offs[None, :] < embedding_dim),
                 other=0.0)

    # 按目标行分组后做 batched atomic add
    for t in range(BLOCK_TOK):
        if tok_start + t < num_tokens:
            row = tl.load(indices_ptr + tok_start + t)
            grad = go[t, :]  # BLOCK_DIM 向量
            # batched atomic add 到 grad_weight[row, dim_start:dim_start+BLOCK_DIM]
            tl.atomic_add(grad_weight_ptr + row * stride_gw_row +
                          dim_offs * stride_gw_dim,
                          grad,
                          mask=dim_offs < embedding_dim)
```

关键实现细节：
- 2D grid 同时并行 token 和 embedding dim；
- `BLOCK_DIM` 对齐硬件向量宽度，保证 atomic 事务高效；
- 内部循环按 token 逐个处理，但每个 token 的更新是向量化的 batched atomic；
- 对重复索引通过 atomic 保证正确性。

## 已知实例

- `exp_embedding_backward_001` — Scheduling（收益 35%）
  - Embedding backward：2D block tiling、软件流水、batched atomic accumulation。大 batch 和大 embedding dim 场景收益明显。重复索引率高时会因 atomic 序列化影响性能。

## 关联知识

- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md)
- 相关 runbook: [Gather/Scatter 内存访问优化：MoE 路由、专家 binning、向量化索引](../memory-sram/traj-gather_scatter-memory_access.md)
- 相关 runbook: [原子操作粒度--连续段 E>=64 用向量 atomic_add，禁止标量循环 atomic](../algorithm-numerics/atomic-granularity.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 主题相关：共享 910b2, 910b3, backward, embedding
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [原子操作粒度--连续段 E>=64 用向量 atomic_add，禁止标量循环 atomic](../algorithm-numerics/atomic-granularity.md) — 同属优化知识，主题相关
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, embedding
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, backward, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
