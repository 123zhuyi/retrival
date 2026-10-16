---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: gather_scatter
dimension: "1.1"
title: Gather/Scatter 内存访问优化聚合
description: 聚合 10 条 gather/scatter 算子在 Ascend 910B 上的 memory access 优化经验，覆盖 MoE 路由、专家 binning、向量化索引、padding 对齐与寄存器缓冲。
tags: [gather_scatter, memory_access, moe, routing, vectorization, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/gather_scatter/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Gather/Scatter 内存访问优化聚合

## 概述

Gather 和 Scatter 是 MoE、推荐系统、稀疏训练中的关键算子。它们的访问模式通常是不规则的：token 或样本按照索引被分发到不同专家/位置。在 Ascend 910B 上，**标量索引循环和随机内存访问是主要瓶颈**。优化核心包括：把索引操作向量化、按专家 binning 减少随机访问、用软件流水隐藏延迟、以及通过 padding 和指针对齐满足硬件事务要求。

本卡聚合 10 条 gather/scatter 的 memory access 经验，收益 20%–35%，可作为 MoE 路由层和稀疏索引操作的优化参考。

## 共同模式

### 1. 向量化索引加载与 Scatter/Gather

原始实现常用标量循环逐个处理索引，导致：
- 每次 load/store 只有一个元素；
- 无法利用合并访问；
- 循环控制开销高。

优化方式：
- 用 `tl.load` 一次性加载 BLOCK 个索引；
- 根据索引计算目标地址，批量执行 gather 或 scatter；
- 用 mask 处理边界。

注意：scatter 若存在索引冲突（多个 token 落到同一位置），需要原子操作或保证输入索引已排序/无冲突。

### 2. Expert Binning 减少随机访问

在 MoE 场景中，token 到专家的映射通常是随机的。若直接按原始顺序 scatter，内存访问会跨越大片地址空间。Expert binning 的优化：
- 预先把 token 按专家 ID 排序或分组；
- 每个 program 处理一个 expert bin，访问连续地址；
- 用 `expert_start_loc` / `expert_capacity` 管理每个 bin 的边界。

这是 MoE scatter/gather 最关键的优化之一。

### 3. 软件流水隐藏内存延迟

Gather/scatter 的索引计算和地址生成通常不耗时，真正的延迟来自全局内存访问。优化方式：
- 预取下一批索引/数据；
- 在 `tl.range` 中设置 `num_stages` 实现软件流水；
- 让地址生成与当前数据加载重叠。

### 4. Padding 与指针对齐

硬件内存事务通常按固定大小（如 32B/64B/128B）执行。如果 gather/scatter 的 block 不对齐，会导致：
- 部分事务浪费；
- 需要额外的 mask 处理。

优化方式：
- 对 expert bin 或 token block 做 padding，使其大小对齐硬件事务；
- 使用 `tl.make_block_ptr` 时设置合适的 boundary_check；
- 在 wgrad scatter 中对齐指针，减少标量地址计算。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_moe_scatter_001` | 35% | MoE scatter：软件流水、向量化 scatter、局部 prefix sum、FP8 scale |
| `exp_scatter_001` | 30% | Scatter：向量化、sub-block tiling、边界 mask、混合精度 |
| `exp_scatter_npu_001` | 35% | MoE routing scatter：binning、向量化 block、寄存器复用 |
| `exp_padded_gather_001` | 30% | Padded gather：buffered writes、列 tiling、动态边界 |
| `exp_gather_flat_001` | 25% | Flat gather：大 block、stride 地址、向量化 mask |
| `exp_gather_001` | 30% | Gather：向量化索引、batch 加载、控制流优化 |
| `exp_moe_gather_001` | 20% | MoE gather：向量化 metadata、循环展开、内存对齐 |
| `exp_scatter_wgrad_moe_001` | 30% | MoE wgrad scatter：expert binning、padding 对齐、向量化归约 |
| `exp_scatter_moe_padded_001` | 30% | MoE padded scatter：expert grouping、向量化、padding 对齐 |
| `exp_moe_binned_scatter_001` | 30% | MoE binned scatter：寄存器缓冲、向量化、动态 bin |

## 代表 Kernel

以下为 `exp_moe_scatter_001` 对应的 kernel，收益 35%，体现了 **MoE scatter + 软件流水 + 向量化 + 局部 prefix sum**：

```python
@triton.jit
def _fwd_kernel_ep_scatter_2(
    input_ptr, output_ptr, indices_ptr,
    expert_start_loc_ptr, expert_capacity,
    n_tokens, hidden_dim,
    stride_in_tok, stride_in_hid,
    stride_out_exp, stride_out_cap, stride_out_hid,
    BLOCK_HID: tl.constexpr, BLOCK_TOK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    for tok_start in range(pid * BLOCK_TOK, n_tokens, num_pid * BLOCK_TOK):
        tok_offsets = tok_start + tl.arange(0, BLOCK_TOK)
        tok_mask = tok_offsets < n_tokens

        # 加载专家索引
        expert_ids = tl.load(indices_ptr + tok_offsets, mask=tok_mask)

        # 局部 prefix sum：计算每个 token 在对应 expert bin 内的位置
        # 实际实现中通常使用 pre-computed expert_start_loc
        positions = tok_offsets - tl.load(expert_start_loc_ptr + expert_ids)

        # 加载输入 token 特征
        hid_offsets = tl.arange(0, BLOCK_HID)
        x = tl.load(input_ptr +
                    tok_offsets[:, None] * stride_in_tok +
                    hid_offsets[None, :] * stride_in_hid,
                    mask=tok_mask[:, None])

        # 计算输出地址：按 (expert, position, hidden) 存储
        out_ptrs = output_ptr +
                   expert_ids[:, None] * stride_out_exp +
                   positions[:, None] * stride_out_cap +
                   hid_offsets[None, :] * stride_out_hid

        tl.store(out_ptrs, x, mask=tok_mask[:, None])
```

关键实现细节：
- grid-stride 外循环让每个 program 处理多个 token block；
- 索引和特征都使用向量化加载；
- `expert_start_loc` 预计算每个 expert 在输出中的起始位置，避免运行时 prefix sum；
- 输出按 `(expert, capacity, hidden)` 布局，便于后续 expert compute。

## 已知实例

- `exp_moe_scatter_001` — Memory Access（收益 35%）
  - MoE scatter：软件流水、向量化 scatter、局部 prefix sum、FP8 scale、内存对齐 padding。
- `exp_scatter_001` — Memory Access（收益 30%）
  - 通用 scatter：向量化内存访问、sub-block tiling、边界 mask、混合精度 compute。
- `exp_scatter_npu_001` — Memory Access（收益 35%）
  - MoE routing scatter：binning 策略、向量化 block 处理、寄存器复用 scale。
- `exp_padded_gather_001` — Memory Access（收益 30%）
  - Padded gather：buffered writes、列 tiling、动态边界处理。
- `exp_gather_flat_001` — Memory Access（收益 25%）
  - Flat gather：大 block 处理、stride-based 地址、向量化 mask。
- `exp_gather_001` — Memory Access（收益 30%）
  - 通用 gather：向量化索引、batch 加载、控制流优化、行级向量化。
- `exp_moe_gather_001` — Memory Access（收益 20%）
  - MoE gather：向量化 metadata 加载、循环展开、内存对齐。
- `exp_scatter_wgrad_moe_001` — Memory Access（收益 30%）
  - MoE wgrad scatter：expert binning、padding 对齐、向量化归约、指针对齐 hint。
- `exp_scatter_moe_padded_001` — Memory Access（收益 30%）
  - MoE padded scatter：expert grouping、向量化内存访问、padding 对齐、sub-block tiling。
- `exp_moe_binned_scatter_001` — Memory Access（收益 30%）
  - MoE binned scatter：寄存器缓冲、向量化内存访问、动态 bin 处理、地址计算优化。

## 关联知识

- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, vectorization
- 对应指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, gather_scatter, moe, routing
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](../../ops/moe-finalize-routing-v2-grad.md) — 主题相关：共享 moe, routing
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](../../ops/moe-finalize-routing-v2.md) — 主题相关：共享 moe
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 主题相关：共享 moe
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 moe
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 主题相关：共享 moe
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](../../ops/moe-init-routing-v2-grad.md) — 主题相关：共享 moe
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](../../ops/moe-init-routing-v2.md) — 主题相关：共享 moe, routing
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](../../ops/moe-init-routing-v3.md) — 主题相关：共享 moe
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](../../ops/moe-init-routing.md) — 主题相关：共享 moe, routing
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 主题相关：共享 moe
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 主题相关：共享 moe
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Activation Scheduling 优化聚合](../compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Scatter Weight Gradient 向量化计算优化](../compute-pipeline/traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, gather_scatter, vectorization
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](../compute-pipeline/traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, gather_scatter
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](../compute-pipeline/traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, moe, vectorization
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](parallel-axis-align.md) — 主题相关：共享 gather_scatter
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, vectorization
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
