---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: conv
dimension: "2.3"
title: Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退
description: 聚合 3 条 Causal Conv1D 在 Ascend 910B 上的 scheduling 经验，覆盖 Prefill 阶段 2D Grid + BTD 布局、Update 阶段 1D Grid + BDT 布局 + 3D 任务分解 + 双缓冲状态更新，以及 continuous batching 下的 pad_slot 早退机制，整体收益 25%–1.6x。
tags: [conv, scheduling, causal_conv1d, btd_layout, bdt_layout, two_d_grid, pad_slot, continuous_batching, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/conv/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退

## 概述

Causal Conv1D 在 Mamba/SSM 类模型中分两个阶段执行：**Prefill**（长序列初始化）和 **Update**（逐 token decode）。两个阶段的数据特征完全相反，导致调度策略也截然相反：
- **Prefill**：`seq_len` 大（512–4096），`D` 也大，但计算沿 T 方向有滑动窗口依赖；
- **Update**：`seq_len=1`，`D` 大，主要瓶颈是权重加载和状态更新的原子性。

本卡 3 条 experience 覆盖了三个关键调度决策：
1. **Prefill 用 2D Grid + BTD 布局 + 仅 D 维 tile 化**，把 D 并行展开以提升 SM 占用率；
2. **Update 用 1D Grid + BDT 布局 + 3D 任务分解 + 双缓冲状态更新**，解决短序列下的矩阵式卷积和状态竞争；
3. **pad_slot 早退机制**，让 continuous batching 中的填充序列立即释放 compute unit。

**核心 trade-off**：
- Prefill 的 BTD 布局对长序列友好，但不能对 T 做 tile 化，否则会破坏跨 token 的滑动窗口寄存器复用；
- Update 的 BDT 布局让 D 维连续，但 D_CHK_SIZE 必须小，否则 T 方向跨步严重；
- 2D Grid 提升占用率，但要求布局与 tile 策略严格匹配；1D Grid + 3D 任务分解更灵活，但任务调度逻辑更复杂；
- 双缓冲状态更新消除 WAR 竞争，但增加一份状态 buffer 和一次 host 同步。

## 共同模式

### 1. Prefill：2D Grid + BTD 布局 + 仅 D 维 tile 化

**问题根因**：Prefill 阶段 `seq_len` 大，若用 1D Grid（每条序列一个 program），单个 program 要处理完整 D 维（如 4096），寄存器压力大且 SM 占用率低。同时 BTD `[B, T, D]` 布局下同一 token 的 D 维元素连续，适合大向量加载。

**收益明显条件**：
- `dim >= 512`，`batch >= 1`；
- Prefill 阶段 `seq_len` 长（512–4096）；
- 不需要在 T 维做任务切分（即每个 program 处理整条序列）。

**实现要点**：
```python
grid = (batch, triton.cdiv(dim, BLOCK_D))
kernel[grid](x, weight, conv_states, ...)

@triton.jit
def kernel(..., BLOCK_D: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
```
- Grid 第二维沿 D 展开，每个 program 负责一个 D-tile 的完整 T 序列；
- BTD 布局保证 D 维连续，单次 load 可拿到 1024 个连续元素；
- 利用滑动窗口寄存器复用跨 T 计算（详见 Causal Conv1D compute/memory access 卡）。

**常见坑**：
- 不要对 T 维做 tile 化，否则会打断跨 tile 的状态传递，需要额外重叠/同步逻辑；
- BLOCK_D 太大可能 spill，需通过遗传搜索或 autotune 确定；
- 必须确认输入是 BTD 布局，否则 2D Grid 的优势会被跨步访问抵消。

### 2. Update：1D Grid + BDT 布局 + 3D 任务分解

**问题根因**：Update 阶段 `seq_len=1`，但 `D` 很大。若用 BTD 布局，短序列下矩阵式卷积的权重/输入加载不连续；BDT `[B, D, T]` 布局让 D 维连续，权重 tile `[D_CHK_SIZE, W]` 可一次 block load。

**收益明显条件**：
- Decode 阶段，`seqlen <= 256`（通常就是 1）；
- `dim` 大（如 4096），`width >= 2`；
- 需要 D 维和 T 维同时 tile 化以获得足够并行度。

**实现要点**：
```python
pid = tl.program_id(0)
total_tasks = batch * NUM_D_CHK * NUM_T_CHK
for task_id in tl.range(pid, total_tasks, tl.num_programs(0)):
    bi = task_id // (NUM_D_CHK * NUM_T_CHK)
    di = (task_id // NUM_T_CHK) % NUM_D_CHK
    ti = task_id % NUM_T_CHK
```
- 1D Grid 通过 strided loop 把三维任务 `(batch, NUM_D_CHK, NUM_T_CHK)` 映射到硬件；
- `D_CHK_SIZE` 取小（如 16），保证 BDT 布局下 T 维连续；
- `T_CHK_SIZE` 取大（如 256），覆盖足够多 token 减少 tile 数量。

**常见坑**：
- BDT 布局下若 `D_CHK_SIZE` 太大，T 方向跨步 = `seq_len`，访问严重不连续；
- 1D Grid 的负载均衡依赖 `total_tasks` 与核心数匹配，任务太少时空载；
- D-tile 和 T-tile 的边界需要分别处理。

### 3. 双缓冲状态更新

**问题根因**：多个 program 并发更新同一条序列的 `conv_state`，直接原地写入会产生 WAR（Write-After-Read）竞争，必须串行化或加锁。

**收益明显条件**：
- Update kernel 中多个 D-tile 同时写入同一 batch 的状态；
- 状态更新与状态读取不需要在同一个 kernel 内完成。

**实现要点**：
- kernel 写入独立的 `conv_state_update` buffer；
- kernel 结束后 host 通过 `conv_state.copy_(conv_state_update)` 一次性同步；
- 读写分离，多个 program 可无竞争地并发写入不同区域。

**常见坑**：
- 增加一份状态 buffer 的内存占用；
- host 同步点必须放在所有 program 完成之后；
- 如果状态更新逻辑可融合到下一个 kernel，可进一步减少同步。

### 4. Host 端预计算负偏移常量

**问题根因**：Update 阶段处理状态更新时，某些 tile 的起始偏移 `t_off` 可能小于 `-(W-1)`，kernel 内用 `tl.maximum` 处理负偏移会生成标量指令。

**收益明显条件**：
- Update kernel 的状态 store 头 tile；
- `seqlen % T_CHK_SIZE` 与 `width` 的关系已知且固定。

**实现要点**：
```python
# host 端预计算，作为 constexpr 传入
ST_STORE_HEAD_TILE_SIZE = (
    width if (seqlen % T_CHK_SIZE) > width
    else (width - seqlen % T_CHK_SIZE) % T_CHK_SIZE
)
```
- 避免 kernel 内动态负偏移计算；
- 编译器可 specialization，减少 scalar 指令。

**常见坑**：
- 公式要覆盖 `seqlen % T_CHK_SIZE == 0` 的情况；
- 该常量只对状态更新的头 tile 有效，不要滥用。

### 5. pad_slot 早退：continuous batching 优化

**问题根因**：动态 batch 中常包含 padding 序列或 beam search 中多条序列共享同一 cache slot。执行这些无效序列会浪费 compute unit，降低有效 SM 占用率。

**收益明显条件**：
- batch 中存在 padding 序列（如 10% 以上）；
- 使用 `cache_indices` 做 batch_id → cache_slot 的间接映射；
- kernel 入口判断成本低。

**实现要点**：
```python
cache_idx = tl.load(cache_indices_ptr + pid_batch)
if cache_idx == pad_slot_id:   # -1
    return                      # 立即释放 compute unit
```
- 仅 1 次 load + 1 次整数比较；
- 释放出的 tile 可被 grid-stride 调度重新分配给其他 tasks。

**常见坑**：
- `return` 必须放在 kernel 最前面，避免已产生副作用；
- 如果同 warp 内部分 program 退出、部分不退出，可能仍有少量 divergence，但通常收益远大于开销；
- `cache_indices` 布局需连续，load 才能合并。

### 6. 布局驱动的 tile size 选择

**问题根因**：同样的 tile size 在不同布局下可能从“合并访问”变成“跨步访问”。

**收益明显条件**：
- 所有 Conv1D 变体；
- 输入布局在编译期已知（BTD 或 BDT）。

**实现要点**：
| 阶段 | 布局 | Grid | D tile | T tile | 原因 |
|---|---|---|---|---|---|
| Prefill | BTD `[B,T,D]` | 2D `(batch, D-tiles)` | 大（1024） | 不切 | D 连续、T 需跨 token 寄存器复用 |
| Update | BDT `[B,D,T]` | 1D + 3D 分解 | 小（16） | 大（256） | D 连续但需避免 T 方向跨步 |

**常见坑**：
- 不要混用布局和 tile 策略（如 BTD + 小 D tile，或 BDT + 大 D tile）；
- 布局转换本身有成本，尽量在模型层面固定布局而非在 kernel 内频繁转置。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_causal_conv1d_update_007` | 30%–40% | Update kernel 整体架构：BDT 布局 + 1D Grid 3D 任务分解 + 双缓冲状态更新 + 小 D_CHK_SIZE。 |
| `exp_causal_conv1d_prefill_003` | 25%–39% / 1.60x | Prefill kernel：2D Grid + BTD 布局 + 仅 D 维 tile 化，遗传搜索验证为全局最优。 |
| `exp_causal_conv1d_prefill_005` | 等于 padding 比例 | Prefill pad_slot 早退：1 load + 1 cmp 跳过无效序列，释放 compute unit 给其他 tasks。 |

## 代表 Kernel

选择 `exp_causal_conv1d_prefill_003` 作为代表，因为它展示了**布局选择、Grid 维度、tile 策略三者如何耦合**，并且该模式可推广到其他“长序列 + 大 dim”的算子。以下为基于该 experience 的 2D Grid Prefill 示意：

```python
@triton.jit
def causal_conv1d_prefill_kernel(
    x_ptr, weight_ptr, out_ptr, conv_state_ptr,
    batch, seqlen, dim, width,
    stride_x_b, stride_x_t, stride_x_d,
    stride_w_d, stride_w_w,
    stride_o_b, stride_o_t, stride_o_d,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim

    # BTD 布局：同一 token 的 D 维连续
    x_base = x_ptr + pid_batch * stride_x_b
    w_base = weight_ptr + pid_d * BLOCK_D * stride_w_d

    # 每个 program 处理完整 T 序列的某个 D-tile
    # 利用滑动窗口寄存器复用：保存前 W-1 个 x 值在寄存器中
    window = tl.zeros((BLOCK_D, width - 1), dtype=tl.float32)

    for t in range(seqlen):
        x_t = tl.load(
            x_base + t * stride_x_t + d_offs[None, :] * stride_x_d,
            mask=d_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # (1, BLOCK_D)

        # 拼接窗口与当前 token
        full = tl.join(window, x_t)  # 示意：实际用 concat 或 reshape
        w = tl.load(
            w_base + tl.arange(0, width)[None, :] * stride_w_w + d_offs[:, None] * stride_w_d,
            mask=d_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_D, width)

        out = tl.sum(full * w, axis=1)  # (BLOCK_D,)
        tl.store(
            out_ptr + pid_batch * stride_o_b + t * stride_o_t + d_offs * stride_o_d,
            out.to(out_ptr.dtype.element_ty),
            mask=d_offs < dim,
        )

        # 滑动窗口移位
        if width > 1:
            window = tl.where(
                tl.arange(0, width - 1)[None, :] < (width - 2),
                window,
                x_t,
            )
```

**关键实现细节**：
- `grid = (batch, triton.cdiv(dim, BLOCK_D))`，第二维把 D 展开到多个 SM；
- BTD 布局保证 `x_t` 的 D 维连续，单次 load 可覆盖整个 BLOCK_D；
- 每个 program 沿 T 循环处理完整序列，保留 `width-1` 历史值在寄存器中实现滑动窗口；
- 不对 T 做 tile 化，避免跨 tile 状态同步。

## 已知实例

- `exp_causal_conv1d_update_007` — Scheduling（收益 30%–40%）
  - Update kernel 整体架构：BDT 布局 + 1D Grid 3D 任务分解 + 双缓冲状态更新 + host 预计算负偏移常量 + 小 D_CHK_SIZE。
- `exp_causal_conv1d_prefill_003` — Scheduling（收益 25%–39%，整体 1.60x）
  - Prefill kernel：2D Grid + BTD 布局 + 仅 D 维 tile 化，遗传搜索验证为全局最优，代表 kernel 见上。
- `exp_causal_conv1d_prefill_005` — Scheduling（收益 ≈ padding 比例）
  - pad_slot 早退：通过 `cache_indices` 判断无效序列，1 load + 1 cmp 立即 return，释放 compute unit。

## 关联知识

- 开发指南: [Conv 算子优化](../../reference/op-class-convolution.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 同属优化知识，主题相关
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 同属优化知识，主题相关
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 causal_conv1d
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 causal_conv1d
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [pad slot 单次检查提前退出，跳过无效填充序列](pad-slot-early-exit.md) — 主题相关：共享 pad_slot
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Causal Conv1D Prefill/Update 跨 Kernel 优化迁移](traj-conv-cross_kernel.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
