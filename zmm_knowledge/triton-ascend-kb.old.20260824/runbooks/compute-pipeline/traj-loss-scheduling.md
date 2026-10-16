---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: loss
dimension: "2.3"
title: 大词表 Entropy Loss Backward 调度融合优化
description: 针对大词表 entropy loss backward，通过 grid-stride 调度、grouped tiling、指针算术优化与 logits 重算融合，将 d_logits/d_hidden/d_weight 合并到一个 kernel，最高提升 35%。
tags: [loss, scheduling, entropy_backward, grid_stride, operator_fusion, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/loss/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 大词表 Entropy Loss Backward 调度融合优化

## 概述

在大词表（vocab_size 数十万到数百万）语言模型中，entropy loss backward 需要计算 `d_logits`、`d_hidden` 和 `d_weight` 三个输出。若拆成多个 kernel，会把巨大的 logits 矩阵反复写入 HBM，带宽成为瓶颈。该组经验通过 **调度优化 + 算子融合**，把 backward 主循环合并为一个 kernel，并在循环内通过 grid-stride、grouped tiling 与编译器友好的指针算术提升 Cube 利用率。

核心收益来源：

- **grid-stride 调度**：启动 grid 固定为 `num_aicores`，kernel 内以 `num_aicores` 为步长遍历所有 MN tile，降低大 grid 的 launch 与调度开销；
- **grouped tiling**：`GROUP_SIZE_M` 把连续 M 维 tile 绑定到同一组，提高 hidden/maximum/accu 等数据的 L2 复用；
- **logits 重算融合**：用 `tl.dot` 在 kernel 内重算 logits，避免从 HBM 读取前向保存的 logits；
- **显式偏移指针算术**：循环内每次用 `base + offset + k * BLOCK_SIZE_K * stride` 计算地址，帮助编译器做别名分析与向量化。

代价是寄存器压力和 atomic contention 增加；当词表或 hidden_size 过大时，需要仔细选择 `BLOCK_SIZE_M/N/K` 与 `GROUP_SIZE_M`。

## 共同模式

### 1. Grid-stride 把逻辑 tile 映射到固定物理核

传统做法把 grid 设为 `num_pid_m * num_pid_n`，每个逻辑 tile 对应一个 program；当词表很大时 grid 尺寸爆炸，调度开销显著。改用 `for pid in range(pid0, total_tiles, num_aicores)` 后，grid 固定为物理 AI Core 数，每个 core 顺序处理多个 tile，显著降低 launch latency。

### 2. Grouped tiling 提高输入复用

`GROUP_SIZE_M` 让同一组 program 先共享 M 维（token）范围，再遍历 N 维（vocab）。这样 `hidden`、`maximum`、`accu`、`d_entropy` 等 M 维数据可以在组内被多个 N tile 复用，减少对 HBM 的重复访问。

### 3. logits 重算替代内存读取

前向保存 logits 需要 `num_tokens × vocab_size` 的内存带宽。backward 中通过 `tl.dot(hidden_tile, weight_tile.trans())` 重算 logits，虽然增加了 Cube 计算量，但避免了大量 HBM 读写，整体更快。

### 4. 循环内显式地址计算

在 K 维循环里，不维护 `hidden_ptrs += BLOCK_SIZE_K * stride` 这种增量指针，而是每次重新计算 `hidden_ptr + offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k + k * BLOCK_SIZE_K * stride_hidden_k`。这种写法让编译器更容易识别每次 load 的基地址与偏移，生成更高效的向量 load。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_entropy_backward_fused_001` | 35% | 将 entropy loss backward 的 d_logits/d_hidden/d_weight 融合到一个 kernel，并通过 grid-stride + grouped tiling 调度，在大词表场景提升 35%。 |
| `exp_entropy_backward_d_weight_001` | 25% | 独立的 d_weight backward kernel，采用 grid-stride、grouped GEMM 调度与 `tl.atomic_add` 累积。 |
| `exp_entropy_backward_001` | 25% | 独立的 d_hidden backward kernel，通过 int32 标签比较避免 NPU int64 标量退化。 |

## 代表 Kernel

`exp_entropy_backward_fused_001` 同时覆盖 grid-stride、grouped tiling、logits 重算与指针算术四种调度模式，因此选为代表 kernel：

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 16,
            },
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_mainloop_MN(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    num_aicores: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    backward mainloop, where d_logits & d_hidden & d_weight are fused
    """
    pid0 = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # num_aicores is the number of cube cores available on the current hardware. When launching the kernel,
    # the grid is set to num_aicores, and inside the kernel a loop is executed with num_aicores as the step size.
    # Compared with setting the grid to num_pid_n * num_pid_m, this approach can reduce the launch and
    # scheduling time caused by a large number of logical cores being dispatched.
    for pid in range(pid0, total_tiles, num_aicores):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_in_group = pid % num_pid_in_group
        valid_in_group = group_size_m * num_pid_n
        if pid_in_group < valid_in_group:
            pid_m = first_pid_m + (pid_in_group % group_size_m)
            pid_n = pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            maximum_ptrs = maximum_ptr + offs_am * stride_maximum
            maximum = tl.load(maximum_ptrs, mask=offs_am < num_tokens, other=0.0)
            accu_ptrs = accu_ptr + offs_am * stride_accu
            accu = tl.load(
                accu_ptrs, mask=offs_am < num_tokens, other=1e-6
            )  # epsilon to avoid division by zero
            accu_rcp = tl.fdiv(1.0, accu)

            d_entropy_ptrs = d_entropy_ptr + offs_am * stride_d_entropy
            d_entropy = tl.load(d_entropy_ptrs, mask=offs_am < num_tokens, other=0.0)
            if reduction == 0:  # none
                d_logprobs_ptrs = d_logprobs_ptr + offs_am * stride_d_logprobs
                d_logprobs = tl.load(
                    d_logprobs_ptrs, mask=offs_am < num_tokens, other=0.0
                )
            elif reduction == 1:  # sum
                d_logprobs = tl.load(d_logprobs_ptr)
                d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
            else:  # mean
                d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
                d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
            d_logprobs = -1 * d_logprobs

            entropy_b_ptrs = entropy_b_ptr + offs_am * stride_entropy_b
            entropy_b = tl.load(entropy_b_ptrs, mask=offs_am < num_tokens, other=0.0)

            hidden_ptrs = hidden_ptr + (
                offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
            )
            weight_ptrs = weight_ptr + (
                offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
            )
            labels_ptrs = labels_ptr + offs_am * stride_labels
            labels = tl.load(labels_ptrs, mask=offs_am < num_tokens, other=0).to(
                tl.int32
            )

            logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                mask_k = offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=mask_k & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight = tl.load(
                    weight_ptrs,
                    mask=mask_k & (offs_bn[:, None] < vocab_size),
                    other=0.0,
                )

                logits = tl.dot(_hidden, _weight.trans(), logits)

                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # scale logits by temperature
            logits *= rcp_temperature

            exp_logits = tl.exp(logits - maximum[:, None])

            mask = (offs_bn + rank * vocab_size)[None, :].to(tl.int32) == labels[
                :, None
            ]
            d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
            d_logits += (
                d_entropy[:, None]
                * (-exp_logits * accu_rcp[:, None])
                * (logits - entropy_b[:, None])
            )

            # scale d_logits by temperature
            d_logits *= rcp_temperature
            d_logits = d_logits.to(tl.bfloat16)
            d_logits_trans = d_logits.trans()

            # loop for d_weight & d_hidden
            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                mask_k = offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K
                mask_am = mask_k & (offs_am[:, None] < num_tokens)
                mask_bn = mask_k & (offs_bn[:, None] < vocab_size)

                # When performing loads inside a loop, directly adding the target offset to the base address
                # each time—rather than maintaining a separate offset variable and incrementing it by BLOCK_SIZE_K * stride_hidden_k
                # on each iteration—makes it easier for the compiler to infer information about the data being loaded,
                # thereby improving the performance of the load operations.
                _hidden = tl.load(
                    hidden_ptr
                    + (
                        offs_am[:, None] * stride_hidden_m
                        + offs_k[None, :] * stride_hidden_k
                    )
                    + k * BLOCK_SIZE_K * stride_hidden_k,
                    mask=mask_am,
                    other=0.0,
                )

                _d_weight = tl.dot(d_logits_trans, _hidden)
                _weight = tl.load(
                    weight_ptr
                    + (
                        offs_bn[:, None] * stride_weight_n
                        + offs_k[None, :] * stride_weight_k
                    )
                    + k * BLOCK_SIZE_K * stride_weight_k,
                    mask=mask_bn,
                    other=0.0,
                )
                tl.atomic_add(
                    d_weight_ptr
                    + offs_bn[:, None] * stride_d_weight_n
                    + offs_k[None, :] * stride_d_weight_k
                    + k * BLOCK_SIZE_K * stride_d_weight_k,
                    _d_weight,
                    mask=mask_bn,
                )

                _d_hidden = tl.dot(d_logits, _weight)

                tl.atomic_add(
                    d_hidden_ptr
                    + offs_am[:, None] * stride_d_hidden_m
                    + offs_k[None, :] * stride_d_hidden_k
                    + k * BLOCK_SIZE_K * stride_d_hidden_k,
                    _d_hidden,
                    mask=mask_am,
                )
```

## 已知实例

- `exp_entropy_backward_fused_001` — Scheduling Optimization（收益 35%）
  - 融合 d_logits/d_hidden/d_weight 的 entropy backward 主循环，采用 grid-stride、grouped tiling、logits 重算与显式指针算术。
  - 对应 kernel: `raw_experiences/extracted_kernels/206-efficient_entropy_backward_kernel_general_mainloop_MN.py`
- `exp_entropy_backward_d_weight_001` — Scheduling Optimization（收益 25%）
  - 仅计算 d_weight 的变体，同样使用 grid-stride 与 grouped GEMM 调度，通过 `tl.atomic_add` 处理跨 token 的 weight 梯度竞争。
  - 对应 kernel: `raw_experiences/extracted_kernels/207-efficient_entropy_backward_kernel_d_weight.py`
- `exp_entropy_backward_001` — Scheduling Optimization（收益 25%）
  - 仅计算 d_hidden 的变体，重点优化 int32 标签比较以避免 NPU int64 标量路径。
  - 对应 kernel: `raw_experiences/extracted_kernels/208-efficient_entropy_backward_kernel_d_hidden.py`

## 关联知识

- 开发指南: [Loss 算子优化](../../reference/op-class-loss.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](../portability/traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 grid_stride
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, operator_fusion, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
