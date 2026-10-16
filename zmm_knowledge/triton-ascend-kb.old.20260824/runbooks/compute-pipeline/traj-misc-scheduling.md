---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: misc
dimension: "2.3"
title: Misc 算子 Scheduling 优化聚合
description: 聚合 7 条 misc 类别算子在 Ascend 910B 上的 scheduling 优化经验，覆盖 block 内局部排序、深度算子融合、投机解码采样、persistent kernel 与对角分块调度。
tags: [misc, scheduling, block_sort, operator_fusion, speculative_decoding, persistent_kernel, diagonal_scheduling, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/misc/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Misc 算子 Scheduling 优化聚合

## 概述

misc 类别汇集了功能各异但调度逻辑值得抽象的小算子：argsort shuffle、Transformer block 融合、投机解码采样、element-wise 乘法、对角分块调度工具、persistent linear kernel。它们的共同点是**计算逻辑不复杂，但调度方式对性能影响巨大**。本卡聚合 7 条 scheduling 经验，收益 20%–40%。

这些经验覆盖的模式包括：
- 块内局部排序与 shuffle；
- 多算子深度融合 + 异构流水线；
- 投机解码中的向量化与串行处理平衡；
- persistent/grid-stride 循环摊还 launch 开销；
- 对角遍历改善 GEMM 局部性。

## 共同模式

### 1. Block 内局部排序 / Shuffle

某些算子（如 argsort、duplicate key shuffle）只需要在 block 内部重排元素，而非全局排序。优化关键是：
- 限制 shuffle 范围在 BLOCK_SIZE 内，利用 SRAM 做临时存储；
- 并行 PRNG 状态管理；
- in-place 或最小额外内存策略。

### 2. 深度算子融合 + 异步流水线

将 Transformer block 中连续出现的 Sigmoid、Norm、RoPE、Matmul 等操作融合到一个 kernel，并通过 `dl.async_task` 把 Cube 和 Vector 任务异步调度。这比单独优化每个算子更有效，因为消除了中间张量的 HBM 往返。

### 3. 投机解码采样的向量化与串行平衡

投机解码中的 rejection/random sample kernel 需要在 block 内做串行比较（确定接受长度），但 load/store 可以高度向量化。优化策略：
- 全局 load/store 用 `tl.arange` 向量完成；
- 块内比较逻辑用短循环 + mask 传播；
- 对 greedy path 做特殊化，减少分支。

### 4. Persistent Kernel / Grid-Stride 摊还 Launch 开销

对于输出 tile 数量远超物理核心数的大矩阵乘或 element-wise 操作，使用 persistent kernel：
- grid size 固定为核心数；
- 每个 program 通过 stride 循环处理多个 tile；
- 显著降低 kernel launch 和调度延迟。

### 5. 对角分块调度工具

`grouped_launch_diagonal` 这类工具函数把 block 索引映射为对角线坐标，改善大 GEMM 的 L2 cache 命中率和 bank 分布。它本身不是算子，但被多个 matmul/attention kernel 复用。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_shuffle_blockwise_001` | 35% | Block 内局部 shuffle/sort，SRAM 内 in-place |
| `exp_fused_transformer_block_001` | 35% | Transformer block 深度融合：Sigmoid+Norm+RoPE+Matmul 异步流水线 |
| `exp_rejection_sample_001` | 40% | 投机解码 rejection 采样：向量化 load + 块内串行比较 |
| `exp_element_mul_001` | 25% | Element-wise 乘法：grid-stride、宽向量化、标量外提、autotune |
| `exp_launch_diagonal_001` | 20% | 对角 block 调度工具：LCM-based stride、条件激活 |
| `exp_expand_serial_001` | Baseline | Expand 操作基线：指出标量循环和 `tl.get_element` 是瓶颈 |
| `exp_linear_persistent_001` | 25% | Linear persistent kernel：grid-stride + 显式转置 + 安全 tiling |

## 代表 Kernel

以下为 `exp_fused_transformer_block_001` 对应的 kernel，体现了 **深度算子融合 + Cube/Vector 异步流水线**：

```python
@triton.jit
def fused_sigmoid_single_norm_rope_matmul_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    batch, seq_len, hidden_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    for m_start in range(pid * BLOCK_M, seq_len * batch, num_pid * BLOCK_M):
        # 加载输入 tile
        x = load_x_tile(...)

        # Cube scope: Matmul
        with dl.async_task(cube):
            matmul_out = tl.dot(x, w)

        # Vector scope: Norm + Sigmoid + RoPE
        with dl.async_task(vector):
            norm_out = rmsnorm(matmul_out)
            sigmoid_out = tl.sigmoid(norm_out)
            rope_out = apply_rope(sigmoid_out, cos, sin)

        store(out_ptr, rope_out)
```

关键实现细节：
- 不同操作显式分派到 Cube/Vector scope；
- 中间结果保留在 on-chip memory，不写 HBM；
- 通过 `dl.async_task` 让两个单元流水线执行。

## 已知实例

- `exp_shuffle_blockwise_001` — Scheduling（收益 35%）
  - Block 内局部 shuffle/sort：并行 PRNG、in-place 内存策略，BLOCK_SIZE 需适配 SRAM。
- `exp_fused_transformer_block_001` — Scheduling（收益 35%）
  - Transformer block 深度算子融合 + 异步流水线调度，需要硬件支持 Cube/Vector 任务分派。
- `exp_rejection_sample_001` — Scheduling（收益 40%）
  - 投机解码 rejection 采样：向量化全局 load、块内串行处理、strided 输出、mask-based 拒绝传播。
- `exp_element_mul_001` — Scheduling（收益 25%）
  - Element-wise 乘法：grid-stride、宽向量化、标量 hoist、autotune。
- `exp_launch_diagonal_001` — Scheduling（收益 20%）
  - 对角 block 遍历工具：基于 LCM 的 stride、条件激活，适用于大 GEMM。
- `exp_expand_serial_001` — Scheduling（收益 Baseline）
  - Expand 操作基线：指出串行循环和 `tl.get_element` 是主要瓶颈，建议改为 scatter/gather。
- `exp_linear_persistent_001` — Scheduling（收益 25%）
  - Linear persistent kernel：grid-stride loop、显式软件转置、边界安全 tiling。

## 关联知识

- 开发指南: [Misc / 融合算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [动态序列长度场景使用 do_not_specialize 降低编译开销](../meta-skills/traj-misc-compilation_opt.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Matmul 硬件适配优化聚合](../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, diagonal_scheduling
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](../portability/traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 persistent_kernel
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, operator_fusion, scheduling
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
