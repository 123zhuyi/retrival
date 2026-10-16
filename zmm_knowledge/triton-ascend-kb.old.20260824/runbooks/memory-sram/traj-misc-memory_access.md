---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: misc
dimension: "1.1"
title: Misc 算子内存访问优化聚合
description: 聚合 13 条 misc 类别算子在 Ascend 910B 上的 memory access 优化经验，覆盖融合、向量化、寄存器复用、边界处理等典型模式。
tags: [misc, memory_access, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/misc/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Misc 算子内存访问优化聚合

## 概述

misc 类别汇集了难以归入单一算子类型的实用 kernel：投机解码的采样与验证、loss/penalty 融合、cache 稀疏更新、归约梯度计算等。这些算子的共同特点是**计算简单、内存访问决定性能**。在 Ascend 910B 上，优化核心通常不是提升 FLOPS，而是减少 HBM 读写次数、把标量循环改写为向量化加载、以及让中间数据尽量驻留寄存器。

本卡覆盖 13 条 memory access 方向的 experience，收益从 25% 到 60% 不等，可作为“带宽受限型小算子”的优化参考模板。

## 共同模式

### 1. 多算子融合，减少 HBM 往返

将逻辑上连续的操作合并到一个 kernel 内，避免把中间结果写回 HBM 再读回。典型场景包括：
- RMSNorm + Sinkhorn + weighted sum 融合；
- frequency/presence/repetition penalty + temperature scaling 融合；
- Q/K/V/Z/B/A split + reshape + cat 融合。

收益来源：每消除一次中间张量的写-读，就省下一次完整 HBM 带宽。对于元素级或轻量归约算子，这往往是最主要的收益。

### 2. 标量循环改写为向量化加载

原始实现常用 Python 级 `for` 循环 + `tl.get_element` 逐个处理 token 或元素。在 SIMT/SIMD 架构上，这会导致：
- 大量标量指令占用 Vector 单元；
- 无法利用合并内存访问（coalesced access）；
- 循环控制开销随序列长度线性增长。

优化方式：用 `tl.arange` + `tl.load/tl.store` 一次性加载一个 block，再用 `tl.where`、比较、累加等向量化指令处理。投机解码中的 token 验证、接受/拒绝采样都是典型案例。

### 3. 中间结果寄存器驻留

对于需要多步计算但中间值立刻被消费的算子（如 speculative decoding 的累积概率、masked fill 的读-改-写），应把中间结果保留在寄存器/SRAM，而不是写到全局内存再读回。这能消除一半以上的内存事务量。

典型案例：
- `_join_prob_reject_sampler_kernel` 把 `cum_probs` / `cum_rands` 全程放在寄存器，避免循环内反复 global load/store；
- `masked_fill` 直接在寄存器计算 `tl.where(mask, value, input)` 后一次性 store。

### 4. `tl.make_block_ptr` 与自动边界 mask

对于 2D/3D 张量的规则切片访问，使用 `tl.make_block_ptr` 配合 `boundary_check` 可以：
- 让编译器生成对齐的向量 load/store；
- 自动处理变长序列的尾部 padding，无需手写 mask；
- 在维度整除时通过 `DIM_IS_MULTIPLE` 等编译期开关进一步消除 mask。

适用条件：访问模式是规则的 block（如 top-k padding、cache store、分组归约），且 stride 已知。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_fuse_tlx_001` | 35% | RMSNorm + Sinkhorn + weighted sum 融合，H 维度分 tile 加载复用 |
| `exp_spec_dec_verify_001` | 50% | 投机解码 token 验证：标量循环改为向量比较 |
| `exp_outer_product_001` | 45% | 外积算子：小 N 时 RHS 一次加载、循环外复用 |
| `exp_cache_store_001` | 30% | 稀疏 cache 更新：3D block 向量写，index 驱动地址广播 |
| `exp_fused_qkvzba_split_001` | 25% | Q/K/V/Z/B/A split + reshape + cat 融合，静态循环展开 |
| `exp_logits_penalty_fusion_001` | 35% | 多种 penalty + temperature 融合为单次 logits 遍历 |
| `exp_topk_pad_001` | 25% | `make_block_ptr` + `boundary_check` 处理变长 top-k padding |
| `exp_grpo_bwd_001` | 25% | GRPO backward：三路线性 logits 归约融合为单次循环 |
| `exp_gn_wb_backward_001` | 25% | GroupNorm weight/bias 梯度：子块策略 + fp32 累加 |
| `exp_spec_dec_reject_001` | 60% | 拒绝采样：cum_prob/cum_rand 寄存器驻留 + 向量接受逻辑 |
| `exp_masked_fill_001` | 25% | 原地读-改-写：省一次 global store + 一次 global load |
| `exp_tv_distance_001` | 25% | 逐行归约：软件流水 + 无分支梯度计算 |
| `exp_speculative_recovery_001` | 30% | 投机恢复：SUB_BLOCK 向量加载 + 原地 buffer 复用 |

## 代表 Kernel

以下为 `exp_spec_dec_reject_001` 对应的 kernel，收益最高（60%），集中体现了**寄存器驻留 + 向量化接受逻辑**两个模式：

```python
@triton.jit
def _join_prob_reject_sampler_kernel(
    output_token_ids_ptr,  # [batch, max_spec_len + 1]
    output_accept_lens_ptr,  # [batch]
    draft_token_ids_ptr,  # [batch, max_spec_len]
    draft_probs_ptr,  # [batch, max_spec_len]
    target_probs_ptr,  # [batch, max_spec_len + 1, vocab_size]
    uniform_random_ptr,  # [batch, max_spec_len]
    cum_probs_ptr,  # [batch, max_spec_len]
    cum_rand_ptr,  # [batch, max_spec_len]
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    # draft
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr = draft_probs_ptr + batch_idx * max_spec_len
    batch_target_probs_ptr = target_probs_ptr + batch_idx * (max_spec_len + 1) * vocab_size
    batch_uniform_random = uniform_random_ptr + batch_idx * max_spec_len

    # cum probs
    batch_cum_probs_ptr = cum_probs_ptr + batch_idx * max_spec_len
    batch_cum_rand_ptr = cum_rand_ptr + batch_idx * max_spec_len

    # output ptr
    batch_output_token_ids_ptr = output_token_ids_ptr + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx

    spec_offset = tl.arange(0, max_spec_len)

    uniform_rand = tl.load(batch_uniform_random + spec_offset)
    draft_token_ids = tl.load(batch_draft_token_ids_ptr + spec_offset)
    draft_probs = tl.load(batch_draft_probs_ptr + spec_offset)
    target_probs = tl.load(batch_target_probs_ptr + spec_offset * vocab_size + draft_token_ids)

    ratio = target_probs / draft_probs

    cum_probs = tl.cumprod(ratio, axis=0)
    cum_probs = tl.clamp(cum_probs, 0, 1)
    cum_rands = tl.cumprod(uniform_rand, axis=0)

    tl.store(batch_cum_probs_ptr + spec_offset, cum_probs)
    tl.store(batch_cum_rand_ptr + spec_offset, cum_rands)

    accept_len = 0
    is_accept = False
    for pos in range(0, max_spec_len):
        if not is_accept:
            index = max_spec_len - pos - 1
            cum_prob = tl.load(batch_cum_probs_ptr + index)
            cum_rand = tl.load(batch_cum_rand_ptr + index)

            if cum_prob >= cum_rand:
                accept_len = index + 1
                is_accept = True

    write_mask = spec_offset < accept_len

    tl.store(batch_output_token_ids_ptr + spec_offset, draft_token_ids, mask=write_mask)
    tl.store(batch_output_accept_lens_ptr, accept_len)
```

可进一步改进点：
- `cum_probs` / `cum_rands` 可以全程保留在向量寄存器，避免 `tl.store` 到 `cum_probs_ptr` / `cum_rand_ptr` 再 `tl.load` 回来；
- 接受长度判断可用向量化比较 `tl.max(tl.where(cum_probs >= cum_rands, spec_offset + 1, 0))` 替代标量循环，把复杂度从 O(max_spec_len) 降到 O(1) SIMD 归约。

## 已知实例

- `exp_fuse_tlx_001` — Memory Access（收益 35%）
  - 融合 RMSNorm、Sinkhorn 和大规模 weighted sum，H 维度按 tile 加载以适配 L1/L2 cache。M 小、H 大的场景收益最明显。
- `exp_spec_dec_verify_001` — Memory Access（收益 50%）
  - 投机解码验证阶段：用向量加载和向量比较替代标量循环 + `tl.get_element`。
- `exp_outer_product_001` — Memory Access（收益 45%）
  - 外积算子：当 N 较小时把 RHS 提到循环外加载，从 O(M×N) 读降到 O(N) 读。
- `exp_cache_store_001` — Memory Access（收益 30%）
  - 稀疏 cache 更新：用 3D block 向量化写 + index 广播生成目标地址。
- `exp_fused_qkvzba_split_001` — Memory Access（收益 25%）
  - 将 Q/K/V/Z/B/A split、reshape、cat 合并为一个 kernel，减少多次 launch 和中间 HBM 读写。
- `exp_logits_penalty_fusion_001` — Memory Access（收益 35%）
  - 把 frequency/presence/repetition penalty 和 temperature scaling 融合为单次 logits 遍历。
- `exp_topk_pad_001` — Memory Access（收益 25%）
  - 用 `make_block_ptr` + `boundary_check` 自动处理变长序列的 top-k padding。
- `exp_grpo_bwd_001` — Memory Access（收益 25%）
  - GRPO backward：把 new/ref/old 三路线性 logits 的 LSE 归约融合为单次循环。
- `exp_gn_wb_backward_001` — Memory Access（收益 25%）
  - GroupNorm weight/bias 梯度：SUB_BLOCK 策略降寄存器压力，累加用 fp32。
- `exp_spec_dec_reject_001` — Memory Access（收益 60%）
  - 投机解码拒绝采样：cum_prob/cum_rand 寄存器驻留，向量化接受逻辑。
- `exp_masked_fill_001` — Memory Access（收益 25%）
  - 原地读-改-写，用 `tl.where` 省一次 global store 和一次 global load。
- `exp_tv_distance_001` — Memory Access（收益 25%）
  - 逐行 TV distance：软件流水隐藏延迟，`tl.where` 实现无分支梯度。
- `exp_speculative_recovery_001` — Memory Access（收益 30%）
  - 投机恢复：SUB_BLOCK 向量加载，NO_DRAFT_PROBS 路径原地复用 target_probs buffer。

## 关联知识

- 开发指南: [Misc / 融合算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Misc 算子 Scheduling 优化聚合](../compute-pipeline/traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [动态序列长度场景使用 do_not_specialize 降低编译开销](../meta-skills/traj-misc-compilation_opt.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](../portability/traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
