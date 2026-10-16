---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: attention
dimension: "2.3"
title: Attention Scheduling 优化聚合
description: 聚合 7 条 attention 在 Ascend 910B 上的 scheduling 优化经验，覆盖 Cube/Vector 异步流水线、grid-stride persistent kernel、变长序列处理与块级状态累积。
tags: [attention, scheduling, flash_attention, lightning_attention, hstu, heterogeneous_pipeline, grid_stride, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/attention/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Attention Scheduling 优化聚合

## 概述

Attention 是 LLM 的内存和计算瓶颈，但不同变体（Flash Attention、Lightning Attention、HSTU、SDPA）的优化重点差异很大。在 Ascend 910B 上，**调度策略往往比 tile size 更能决定性能**：如何分配 Cube/Vector 任务、如何处理变长序列、如何持久化 kernel 覆盖大行数、如何利用多缓冲隐藏延迟。本卡聚合 7 条 attention scheduling 经验，收益 15%–55%。

这些经验的共同主题：
- 把 attention 拆分为可并行的块级计算；
- 用 Cube 做矩阵乘，用 Vector 做 mask/softmax/scale；
- 对长序列使用 grid-stride 或 persistent kernel；
- 对线性 attention / HSTU 等变体做块级状态累积。

## 共同模式

### 1. Cube/Vector 异构异步流水线

标准 Flash Attention 的计算流程是：QK^T（Cube）→ softmax（Vector）→ PV（Cube）。在 Ascend 上，如果这两个阶段串行执行，Cube 会等待 Vector 完成 softmax。优化方案：
- 用 `dl.async_task` 把 Cube 任务和 Vector 任务显式分派；
- 通过 workspace buffer 传递中间结果，避免 HBM 往返；
- 让小 tile 的 Cube 计算与 Vector 的 softmax 重叠。

这是 Ascend attention kernel 最常见的优化模式。

### 2. Grid-Stride / Persistent Kernel

当序列长度很大、block 数量远超硬件核心数时，每个 program 只处理一个 block 会导致：
- launch 开销被放大；
- 核心利用率受 grid size 限制；
- 负载不均衡。

Grid-stride 或 persistent kernel 让一个 program 顺序处理多个 KV block：
- grid size 固定为核心数；
- 内层循环按 stride 领取任务；
- 适合 Flash Attention forward/backward 和 SDPA backward。

### 3. 块级状态累积

对于 Lightning Attention、HSTU、线性 attention 等变体，attention 可以表示为递推状态更新。调度关键是：
- 每个 block 先计算局部状态；
- 处理跨 block 的传递状态（类似 prefix scan）；
- 状态矩阵常驻 SRAM/寄存器，减少 HBM 访问。

### 4. 变长序列处理

真实 batch 中的序列长度不同，直接按最大长度 padding 会浪费计算。优化方式：
- 使用 `cu_seqlens` 描述实际长度；
- program ID 映射到实际 token 而非 padding 位置；
- 对 HSTU 等 ragged tensor 场景使用 binning 或 block mask。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_lightning_attn_001` | 45% | Lightning Attention：Cube/Vector 异步、workspace 复用、块级状态累积 |
| `exp_hstu_attn_sched_001` | 15% | HSTU attention：静态任务调度、ragged tensor、对称 compute dispatch |
| `exp_fused_attn_preprocess_001` | 35% | Attention 预处理融合：Matmul + Norm + RoPE 异构流水线 |
| `exp_attn_ascend_dual_pipeline_001` | 55% | Ascend 双流水线：Cube/Vector 异步、多级 workspace、长序列 |
| `exp_hstu_attn_bwd_001` | 25% | HSTU attention backward：activation 重计算、块级累加、向量化 mask |
| `exp_flash_attn_gridstride_001` | 35% | Flash Attention：grid-stride factor=20、软件流水、block_ptr |
| `exp_sdpa_bwd_kv_001` | 25% | SDPA backward KV：block tiling、多缓冲、SRAM 梯度累加 |

## 代表 Kernel

以下为 `exp_attn_ascend_dual_pipeline_001` 对应的 kernel，收益 55%，体现了 **Cube/Vector 双流水线 + 多级 workspace + 长序列优化**：

```python
@triton.jit
def _attn_fwd_split_cv(
    q_ptr, k_ptr, v_ptr, o_ptr,
    workspace_ptr,
    batch_size, n_heads, n_ctx, d_head,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    num_pid_m = tl.num_programs(0)

    # 每个 program 负责一个 (batch, head) 内的多个 M block
    for start_m in range(pid_m * BLOCK_M, n_ctx, num_pid_m * BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        # 加载 Q tile
        q = tl.load(q_ptr +
                    tl.arange(0, batch_size)[:, None, None] * stride_qb +
                    tl.arange(0, n_heads)[None, :, None] * stride_qh +
                    offs_m[None, None, :] * stride_qt +
                    offs_d[None, None, :] * stride_qd,
                    mask=(offs_m < n_ctx)[None, None, :])

        # Cube scope: Q @ K^T
        with dl.async_task(cube):
            acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for start_n in range(0, n_ctx, BLOCK_N):
                k = load_k_tile(...)
                s = tl.dot(q, tl.trans(k), acc=acc_s)
                store_to_workspace(workspace_ptr, s, start_m, start_n)

        # Vector scope: softmax over N per row
        with dl.async_task(vector):
            for start_n in range(0, n_ctx, BLOCK_N):
                s = load_from_workspace(workspace_ptr, start_m, start_n)
                m_ij = tl.max(s, axis=1)
                # ... online softmax update ...
                p = tl.exp(s - m_ij[:, None])
                store_to_workspace(workspace_ptr, p, start_m, start_n)

        # Cube scope: P @ V
        with dl.async_task(cube):
            acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            for start_n in range(0, n_ctx, BLOCK_N):
                p = load_from_workspace(workspace_ptr, start_m, start_n)
                v = load_v_tile(...)
                acc_o = tl.dot(p, v, acc=acc_o)
            tl.store(o_ptr + ..., acc_o.to(o_ptr.dtype.element_ty))
```

关键实现细节：
- 用 `dl.async_task(cube)` 和 `dl.async_task(vector)` 显式分派两类任务；
- workspace buffer 在 SRAM/L1 级别传递中间结果，避免 HBM 往返；
- 长序列下 Cube 和 Vector 可以流水线重叠；
- grid-stride 外循环让一个 program 处理多个 M block。

## 已知实例

- `exp_lightning_attn_001` — Scheduling（收益 45%）
  - Lightning Attention：Ascend `dl` 方言、Cube/Vector 异步同步、workspace 复用、块级状态累积。
- `exp_hstu_attn_sched_001` — Scheduling（收益 15%）
  - HSTU attention：静态任务调度、ragged tensor 处理、对称 compute dispatch、block_ptr mask 加载。
- `exp_fused_attn_preprocess_001` — Scheduling（收益 35%）
  - Attention 预处理融合：Matmul + Norm + RoPE 通过 `dl.async_task` 调度到 Cube/Vector。
- `exp_attn_ascend_dual_pipeline_001` — Scheduling（收益 55%）
  - Ascend 双流水线 attention：Cube/Vector 异步执行、多级 workspace buffering、长序列优化，代表 kernel 见上。
- `exp_hstu_attn_bwd_001` — Scheduling（收益 25%）
  - HSTU attention backward：activation 重计算、融合 activation backward、块级累加、向量化 mask。
- `exp_flash_attn_gridstride_001` — Scheduling（收益 35%）
  - Flash Attention：grid-stride factor=20、软件流水、自适应累加器存储、block_ptr 抽象。
- `exp_sdpa_bwd_kv_001` — Scheduling（收益 25%）
  - SDPA backward KV：block tiling、多缓冲流水、SRAM 梯度累加、混合精度。

## 关联知识

- 开发指南: [Attention 算子优化](../../reference/op-class-attention.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 grid_stride
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
