---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: matmul
dimension: "4.1"
title: Matmul 硬件适配优化聚合
description: 聚合 4 条 matmul 在 Ascend 910B 上的 hardware adaptation 经验，覆盖 wavefront/对角调度、dl 编译 hint、Matmul+Norm+RoPE 异构融合与静态多核分布。
tags: [matmul, hardware_adaptation, ascend, wavefront, diagonal_scheduling, dl_hint, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/matmul/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Matmul 硬件适配优化聚合

## 概述

Matmul 在 Ascend 910B 上的硬件适配，核心问题不是算法，而是**如何让 block 调度匹配内存层次结构、如何利用 `dl` 编译 hint 减少 padding、以及如何让 Cube 和 Vector 单元协同工作**。本卡聚合 4 条 hardware adaptation 经验，覆盖 wavefront/对角调度、`dot_pad_only_k` hint、Matmul+RMSNorm+RoPE 异构融合等模式，收益 25%–35%。

这些经验的共同主题：
- 通过调度策略改善 L2 cache 命中率和 bank 访问分布；
- 用硬件专用 hint 减少无效 padding 和指令开销；
- 把 Matmul 与相邻的 Norm/RoPE 操作映射到不同硬件单元。

## 共同模式

### 1. Wavefront / 对角 Block 调度

传统行主序调度会让多个 core 同时竞争矩阵 A 的同一行和 L2 的同一 bank。Wavefront 或对角调度把相邻 program ID 映射到输出矩阵的对角线附近：
- core 访问的 A/B 数据分散到不同 bank；
- 沿对角线推进时，B 的活跃块可被复用；
- 大矩阵、带宽受限场景收益最明显。

关键参数：`BLOCK_TRESHHOLD` 或类似的阈值，只在 M/N block 数足够大时启用对角调度。

### 2. `dot_pad_only_k` 编译 Hint

Ascend Cube 单元要求矩阵乘输入在 K 维度对齐。默认编译器可能同时在 M/N/K 三个维度 padding，造成：
- 额外的零值数据移动；
- 更大的 tile 占用更多 SRAM；
- 边界处理指令增加。

通过 `tl.compile_hint(x, "dot_pad_only_k")` 或 `dl.compile_hint(...)`，显式告诉编译器**只在 K 维度 padding**，保持 M/N 不变。这是 Ascend matmul 最常用的硬件适配 hint。

### 3. Matmul + Norm + RoPE 异构融合

Transformer decoder 中常见 Matmul 后接 RMSNorm 和 RoPE。这三个操作的特性：
- Matmul：计算密集，适合 Cube；
- RMSNorm：归约 + 元素级，适合 Vector；
- RoPE：元素级旋转，适合 Vector。

通过 `dl.async_task` 把 Matmul 放在 Cube scope，Norm/RoPE 放在 Vector scope，可以让两个单元流水线执行，中间结果保留在 on-chip memory。

### 4. 静态多核分布

对于 shape 固定的场景，可以静态地把 block 分配到 24 个 AI core：
- 避免动态调度的开销；
- 通过 `tl.num_programs` 和 `tl.program_id` 实现简单的 stride 映射；
- 适合推理中 batch/head 维度固定的 kernel。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_matmul_ascend_wavefront_001` | 35% | Wavefront 对角调度 + 硬件对齐 + autotune block shape |
| `exp_fused_matmul_rms_rope_001` | 25% | Matmul + RMSNorm + RoPE 异构异步融合 |
| `exp_rope_matmul_001` | 35% | RoPE + Matmul 融合，dot_pad_only_k，对角调度，batch/head 合并处理 |
| `exp_matmul_ascend_001` | 35% | Ascend 静态多核分布 + grouped_launch_diagonal + K-only padding hint |

## 代表 Kernel

以下为 `exp_matmul_ascend_wavefront_001` 对应的 kernel，收益 35%，体现了 **wavefront 对角调度 + 硬件对齐 hint + autotune**：

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_block_m * num_block_n

    # wavefront / 对角调度：让相邻 pid 访问不同 bank
    for block_idx in range(pid, total_blocks, num_pid):
        task_m_idx, task_n_idx = grouped_launch_diagonal(
            block_idx, num_block_m, num_block_n, BLOCK_TRESHHOLD
        )
        offs_am = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = task_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            tl.compile_hint(a, "dot_pad_only_k")
            tl.compile_hint(b, "dot_pad_only_k")
            accumulator = tl.dot(a, b, acc=accumulator)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

关键实现细节：
- `grouped_launch_diagonal` 把 block 索引映射到对角线坐标；
- `BLOCK_TRESHHOLD` 控制何时启用对角调度；
- `dot_pad_only_k` 同时作用于 A 和 B；
- grid-stride 循环让一个 program 处理多个 block。

## 已知实例

- `exp_matmul_ascend_wavefront_001` — Hardware Adaptation（收益 35%）
  - Wavefront 对角调度策略、硬件特定对齐、autotune block shape、静态多核分布。
- `exp_fused_matmul_rms_rope_001` — Hardware Adaptation（收益 25%）
  - Matmul + RMSNorm + RoPE 异构并行，通过 `dl.async_task` 调度 Cube/Vector。
- `exp_rope_matmul_001` — Hardware Adaptation（收益 35%）
  - RoPE 与 Matmul 融合，使用 `dot_pad_only_k`、对角 block 调度、batch/head 合并处理。
- `exp_matmul_ascend_001` — Hardware Adaptation（收益 35%）
  - Ascend 静态多核分布 + `grouped_launch_diagonal` + K-only padding hint + 原地累加策略。

## 关联知识

- 开发指南: [Matmul / GEMM 算子优化](../../reference/op-class-matmul.md)
- 相关 runbook: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md)
- 相关 runbook: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Matmul Compute 优化聚合](../compute-pipeline/traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Matmul Scheduling 优化聚合](../compute-pipeline/traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../compute-pipeline/traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Misc 算子 Scheduling 优化聚合](../compute-pipeline/traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, diagonal_scheduling
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Attention 硬件适配优化聚合](traj-attention-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend, hardware_adaptation
- 相关主题: [Ascend 硬件资源适配规则聚合](traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation

<!-- okf:related:end -->
