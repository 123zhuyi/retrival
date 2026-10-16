---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: matmul
dimension: "2.3"
title: Matmul Scheduling 优化聚合
description: 聚合 19 条 matmul 在 Ascend 910B 上的 scheduling 优化经验，覆盖异构流水线、分组 GEMM 负载均衡、对角/swizzle 分块、K-only padding 与小矩阵特化。
tags: [matmul, scheduling, gemm, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/matmul/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Matmul Scheduling 优化聚合

## 概述

Matmul/GEMM 是 Transformer 和 LLM 的算力核心，但其性能不仅取决于 tile size，更取决于**如何把任务映射到 Ascend 的 AI Core、如何隐藏内存延迟、以及如何与相邻算子融合**。本卡聚合 19 条 scheduling 方向的 experience，覆盖从单矩阵乘到 Grouped GEMM、从预填充到后投影的多种场景。

这些经验的共同优化目标：
- 提高 AI Core 利用率（减少空闲、均衡负载）；
- 提升 L2/L1 cache 命中率（改善数据局部性）；
- 减少 kernel launch 和 HBM 往返（融合相邻算子）；
- 适配 Ascend Cube 单元的 padding 与精度约束。

## 共同模式

### 1. Cube/Vector 异构异步流水线

在 Transformer decoder 层中，Matmul 后面往往紧跟 RMSNorm、RoPE、SiLU 等元素级操作。Ascend 的 Cube 单元擅长矩阵乘，Vector 单元擅长元素级操作。通过 `dl.async_task` 把两类计算显式分发到不同硬件单元，可以让它们并行或流水线执行，避免 Cube 等待 Vector 完成。

典型融合：
- Matmul + RoPE
- Matmul + RMSNorm
- Matmul + RMSNorm + Sigmoid

收益来源：中间张量不必写回 HBM，且两个单元的执行时间可以重叠。

### 2. Grouped GEMM 动态负载均衡

当一批矩阵的 M 或 K 维度差异较大时，静态 block-to-core 映射会导致“长尾”——某些 core 处理大矩阵时，其他 core 已经空闲。Grouped GEMM 的解决方案是：
- 在 kernel 内维护一个全局或 per-group 的 block 计数器；
- 每个 core 通过 `while cur_block < total_blocks` 动态领取任务；
- 任务粒度以 output block（BLOCK_M × BLOCK_N）为单位。

这样即使不同 group 的计算量差异很大，所有 core 也能保持忙碌，直到最后几个 block。

### 3. 对角 / Swizzle 分块提升 L2 局部性

传统行主序分块会让多个 core 同时访问矩阵 A 的同一行，造成 L2 bank 冲突；同时如果矩阵 B 很大，水平分块会导致 B 被重复加载，超出 L2 cache 容量。

对角分块（diagonal tiling）或 swizzle 映射把相邻 program ID 映射到输出矩阵的对角线附近：
- 各 core 访问的 A/B 数据分散到不同 bank；
- 沿对角线推进时，B 的活跃块可被相邻 core 复用；
- 大矩阵场景下 L2 cache 命中率显著提升。

适用条件：M、N 方向 block 数均较多（通常 ≥ 8×8），小矩阵场景收益会被调度开销抵消。

### 4. `dot_pad_only_k` 编译 hint

Ascend Cube 单元对矩阵乘的输入数据布局有对齐要求。默认情况下，编译器可能需要在 M、N、K 三个维度都进行 padding，带来额外的数据和指令开销。通过 `tl.compile_hint(tensor, "dot_pad_only_k")` 或 `dl.compile_hint(...)`，可以显式告诉编译器**只在 K 维度 padding**，避免 M/N 维度的无效计算和内存访问。

这是 Ascend 特化优化中最常见的 hint 之一，在 19 条 experience 中反复出现。

### 5. 小矩阵与精度特化

对于小矩阵（M、N、K ≤ 64），使用大 tile（128×128）会导致边界 mask 开销高、计算单元填充不足。此时应采用：
- 小 tile（如 16×16×16）；
- 浅层流水线（num_stages=3），避免寄存器溢出；
- `allow_tf32=False` 保证 FP32 累加精度。

收益通常 15% 左右，但对小 shape 的 kernel 这是必要的基线优化。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_matmul_small_tile_001` | 15% | 小矩阵使用 16×16×16 tile，allow_tf32=False，num_stages=3 |
| `exp_grouped_matmul_bkmajor_001` | 35% | Grouped GEMM，B 为 K-major，dot_pad_only_k，动态负载均衡 |
| `exp_matmul_qn_001` | 25% | Attention 中 Q_nope×W_uk，静态 block 分配 + dot_pad_only_k |
| `exp_fused_norm_rope_matmul_001` | 35% | Matmul + Norm + RoPE 异构异步融合 |
| `exp_fused_matmul_rope_001` | 25% | Matmul + RoPE 融合，减少中间 Q/K HBM 写回 |
| `exp_fused_matmul_norm_sigmoid_001` | 30% | Matmul + RMSNorm + Sigmoid 融合 |
| `exp_mm_grouped_reorder_001` | 25% | GROUP_M 重排提升 A 矩阵 L2 命中率 |
| `exp_addmm_ascend_001` | 25% | addmm 动态 autotune + allow_tf32=False |
| `exp_matmul_ascend_diag_001` | 25% | 对角分块 + dot_pad_only_k，NUM_CORES=24 |
| `exp_grouped_gemm_001` | 35% | K-grouped GEMM 动态工作窃取 + dot_pad_only_k |
| `exp_matmul_swizzle_hardware_hint_001` | 25% | block swizzling + persistent kernel + dot_pad_only_k |
| `exp_grouped_gemm_bNmajor_001` | 35% | Grouped GEMM，B 为 N-major，动态负载均衡 |
| `exp_partial_matmul_dynamic_001` | 30% | partial matmul 动态 persistent 调度 + 对角遍历 |
| `exp_fused_matmul_rms_001` | 25% | Matmul + RMSNorm 异构异步融合 |
| `exp_grouped_matmul_001` | 25% | K-grouped matmul 动态 block 分配 |
| `exp_bmm_grouped_001` | 25% | BMM grouped scheduling，GROUP_M 提升 A 复用 |
| `exp_fused_matmul_rmsnorm_001` | 25% | Matmul + RMSNorm KV-cache 融合 |
| `exp_matmul_diag_tiling_001` | 25% | 8×8 对角 tiling，缓解 L2 bank 冲突与容量瓶颈 |
| `exp_grouped_gemm_bkmajor_001` | 35% | Grouped GEMM，B K-major，工作窃取调度 |

## 代表 Kernel

以下为 `exp_grouped_matmul_bkmajor_001` 对应的 kernel，收益 35%，集中体现了 **Grouped GEMM 动态负载均衡 + K-major 布局 + `dot_pad_only_k`** 三个核心模式：

```python
@triton.autotune(configs=get_autotune_config(), key=["N", "K"])
@triton.jit
def _m_grouped_matmul_bKmajor_kernel(
    A,
    B,
    C,
    group_size_ptr,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    total_cores = tl.num_programs(axis=0)
    core_idx = tl.program_id(axis=0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    last_count = 0
    group_start = 0
    group_end = 0

    for group_idx in range(num_groups):
        m = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + m
        num_block_m = tl.cdiv(m, BLOCK_M)
        cur_count = last_count + num_block_m * num_block_n
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        for cur_block in range(cur_block, cur_count, total_cores):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                cur_block - last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD
            )
            offs_am = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_bn = (group_idx * N + task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs_base = A + (offs_am[:, None] * K + offs_k[None, :])
            b_ptrs_base = B + (offs_bn[:, None] * strideBN + offs_k[None, :])
            msk_m = offs_am < group_end
            msk_n = offs_bn < (group_idx + 1) * N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a_ptrs = a_ptrs_base + k * BLOCK_K
                b_ptrs = b_ptrs_base + k * BLOCK_K
                a = tl.load(
                    a_ptrs,
                    mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0,
                )
                tl.compile_hint(a, "dot_pad_only_k")
                b = tl.load(
                    b_ptrs,
                    mask=msk_n[:, None] and (offs_k[None, :] < (K - k * BLOCK_K)),
                    other=0.0,
                )
                b = tl.trans(b)
                tl.compile_hint(b, "dot_pad_only_k")
                accumulator = tl.dot(a, b, acc=accumulator)

            c = accumulator.to(C.dtype.element_ty)
            offs_cm = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_cn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < group_end) and (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
        last_count = cur_count % total_cores
        group_start = group_end
```

关键实现细节：
- 外层按 group 遍历，内层用 `range(cur_block, cur_count, total_cores)` 实现动态任务领取；
- `grouped_launch_diagonal` 把 block 索引映射为对角线坐标，兼顾负载均衡与 L2 局部性；
- B 为 K-major，加载后显式 `tl.trans(b)` 以匹配 `tl.dot` 的布局要求；
- `dot_pad_only_k` 同时作用于 A 和 B，减少 padding 开销。

## 已知实例

- `exp_matmul_small_tile_001` — Scheduling（收益 15%）
  - 小矩阵（M,N,K ≤ 64）使用 16×16×16 tile，禁用 TF32，3 阶段流水线，避免大 tile 的边界 mask 和填充不足。
- `exp_grouped_matmul_bkmajor_001` — Scheduling（收益 35%）
  - B 为 K-major 的 Grouped GEMM，动态负载均衡 + dot_pad_only_k，代表 kernel 见上。
- `exp_matmul_qn_001` — Scheduling（收益 25%）
  - Attention 中 Q_nope × W_uk 计算，静态 block 分配，head 维度在 kernel 内迭代，减少 launch 开销。
- `exp_fused_norm_rope_matmul_001` — Scheduling（收益 35%）
  - Transformer decoder 层中 Matmul + RMSNorm + RoPE 的异构异步融合，Cube 负责 matmul，Vector 负责 norm/rope。
- `exp_fused_matmul_rope_001` — Scheduling（收益 25%）
  - LLaMA/ChatGLM 风格 Matmul + RoPE 融合，避免 Q/K 写回 HBM。
- `exp_fused_matmul_norm_sigmoid_001` — Scheduling（收益 30%）
  - Matmul + RMSNorm + Sigmoid 三重融合，使用 dl.async_task 调度 Cube/Vector。
- `exp_mm_grouped_reorder_001` — Scheduling（收益 25%）
  - 通过 GROUP_M 重排 program ID，提升 A 矩阵在 L2 中的复用率。
- `exp_addmm_ascend_001` — Scheduling（收益 25%）
  - addmm 动态 autotune 选择 tile size，allow_tf32=False，支持非连续 stride。
- `exp_matmul_ascend_diag_001` — Scheduling（收益 25%）
  - 对角分块 + dot_pad_only_k，M/N 足够大时启用，否则回退顺序分块。
- `exp_grouped_gemm_001` — Scheduling（收益 35%）
  - K-grouped GEMM，动态工作窃取，A 显式转置以匹配硬件 dot 指令。
- `exp_matmul_swizzle_hardware_hint_001` — Scheduling（收益 25%）
  - block swizzling + persistent kernel + dot_pad_only_k，改善大 GEMM 的 L2 访问模式。
- `exp_grouped_gemm_bNmajor_001` — Scheduling（收益 35%）
  - B 为 N-major 的 Grouped GEMM，动态负载均衡，accumulator 用 fp32。
- `exp_partial_matmul_dynamic_001` — Scheduling（收益 30%）
  - partial matmul 动态 persistent 调度，支持 B 的子切片，避免内存重组。
- `exp_fused_matmul_rms_001` — Scheduling（收益 25%）
  - Matmul + RMSNorm 融合，Cube/Vector 异步执行。
- `exp_grouped_matmul_001` — Scheduling（收益 25%）
  - K-grouped matmul，动态 block 分配，B N-major 布局优化。
- `exp_bmm_grouped_001` — Scheduling（收益 25%）
  - BMM grouped scheduling，GROUP_M 提升 A 矩阵 L2 复用。
- `exp_fused_matmul_rmsnorm_001` — Scheduling（收益 25%）
  - Matmul + RMSNorm KV-cache 融合，适配动态 seq_len。
- `exp_matmul_diag_tiling_001` — Scheduling（收益 25%）
  - 8×8 对角 tiling，缓解大矩阵场景的 L2 bank 冲突与容量瓶颈。
- `exp_grouped_gemm_bkmajor_001` — Scheduling（收益 35%）
  - Grouped GEMM，B K-major，工作窃取调度，适用于变长 batch。

## 关联知识

- 开发指南: [Matmul / GEMM 算子优化](../../reference/op-class-matmul.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, gemm
- 对应指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 相关主题: [Gemm Basic Skeleton](../../reference/skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 gemm
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Matmul 硬件适配优化聚合](../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 gemm
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
