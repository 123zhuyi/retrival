---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: attention
dimension: "4.1"
title: Attention 硬件适配优化聚合
description: 聚合 5 条 attention 在 Ascend 910B 上的 hardware adaptation 经验，覆盖 HIVM 自动均衡、序列分片、Paged KV、稀疏索引 SIMD 与多缓冲流水。
tags: [attention, hardware_adaptation, ascend, hivm, paged_kv, sparse_attention, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/attention/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Attention 硬件适配优化聚合

## 概述

Attention 在 Ascend 910B 上的高效实现，除了算法设计外，很大程度上取决于**对硬件原语的适配程度**。Ascend 提供了 HIVM（HiAI VM）、`tle`/`cube1` 等专有扩展，以及 Cube/Vector 异构执行能力。本卡聚合 5 条 attention 的 hardware adaptation 经验，覆盖 split-decode attention、GQA paged attention、稀疏 flash attention 等场景，收益 30%–60%。

这些经验的共同目标：
- 把计算映射到最适合的硬件单元（Cube 做矩阵乘，Vector 做 softmax/mask）；
- 通过序列分片和多缓冲流水隐藏内存延迟；
- 利用 Ascend 专有指令加速稀疏/分页访问；
- 通过编译器自动均衡（HIVM auto balance）减少手动调优负担。

## 共同模式

### 1. 序列分片（Sequence Partitioning）

长序列 attention 的内存和计算量都很大。把序列沿长度维度切分为多个片段（如 SEQ_LEN // 2），每个 program 处理一个片段，可以：
- 降低单个 kernel 的寄存器/SRAM 压力；
- 让多个核心并行处理不同片段；
- 与 up/down kernel 配合覆盖完整序列。

典型应用：SDA (split-decode attention)、Flash Attention bwd KV。

### 2. HIVM 自动均衡

HiAI VM 是 Ascend 的运行时抽象，提供了自动负载均衡机制。优化经验表明：
- 把循环边界和任务粒度暴露给 HIVM，比手动硬编码 core 分配更稳健；
- 在多缓冲流水场景下，HIVM auto balance 可以减少核心空闲；
- 需要避免过度细粒度的同步，否则会抵消自动均衡的收益。

### 3. Paged KV Cache 迭代

对于 GQA/MQA 等长上下文推理，KV cache 通常按页（page）存储。优化关键是：
- 用 `bind_sub_block` 等原语把 Vector 核心分组，并行处理多个 head；
- 按 page 迭代加载 KV，而非按连续内存；
- 减少因物理不连续导致的地址计算开销。

### 4. 稀疏索引 SIMD

稀疏 Flash Attention 只计算 Top-K 关注的 KV 块。Ascend 上可通过 `tle.index_select_simd` 等 SIMD 索引原语，高效地从稀疏索引中 gather KV。这比标量索引快得多，但要求：
- 索引预计算好（如通过 Top-K 或 block-sparse mask）；
- workspace 布局对齐 Cube 单元要求；
- 在线 softmax 保持稳定。

### 5. 多缓冲流水

在 attention backward 或 split-decode 中，需要反复加载 Q/K/V 并计算梯度。多缓冲（multibuffer）流水可以：
- 在当前计算 tile 的同时预取下一个 tile；
- 通过 `tl.range(..., num_stages=N)` 或手动 double buffer 实现；
- 显著降低高内存延迟场景下的 stall。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_attn_sda_001` | 35% | Split-decode attention：Ascend HIVM、Cube loops、多缓冲、混合精度 |
| `exp_attn_bwd_q_down_001` | 35% | SDA bwd Q down：HiAI VM、多缓冲、序列分片 |
| `exp_gqa_paged_npu_001` | 45% | GQA paged attention：bind_sub_block、Cube/Vector 流水、page 迭代 |
| `exp_sparse_flash_attn_prefill_001` | 60% | 稀疏 Flash Attention prefill：sparse SIMD、cube1、online softmax、workspace tiling |
| `exp_flashattn_bwd_kv_ascend_001` | 30% | Flash Attention bwd KV：硬件流水线均衡、多缓冲、非对称 tiling、序列分片 |

## 代表 Kernel

以下为 `exp_sparse_flash_attn_prefill_001` 对应的 kernel，收益 60%，集中体现了 **稀疏索引 SIMD + Cube 加速 + online softmax + workspace tiling**：

```python
@triton.jit(do_not_specialize=["T1", "T2", "S1", "S2"])
def sparse_flash_attention_prefill_kernel(
    Q_ptr, K_ptr, V_ptr, Indices_ptr, O_ptr,
    Wksp_K_ptr, Wksp_K2_ptr, Wksp_qk_ptr,
    Wksp_score_ptr, Wksp_sv_ptr,
    stride_qt1, stride_qn2, stride_qn1, stride_qd,
    stride_it1, stride_in2, stride_ik,
    stride_ot1, stride_on1, stride_wod,
    stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
    stride_wqkt1, stride_wqkn1, stride_wqks2,
    stride_wscoret1, stride_wscoren1, stride_wscores2,
    stride_wsvt1, stride_wsvn1, stride_wsvd,
    scale,
    T1, N1: tl.constexpr, T2, G: tl.constexpr,
    D_qk: tl.constexpr, D_v: tl.constexpr, K: tl.constexpr,
    B: tl.constexpr, S1, S2,
    bs1_per_core,
    BLOCK_G: tl.constexpr, BLOCK_SBS: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    t1_start = pid * bs1_per_core
    t1_end = tl.minimum(T1, (pid + 1) * bs1_per_core)

    for idx_t1 in range(t1_start, t1_end):
        idx_b = idx_t1 // S1
        cur_s2 = S2
        beg_s2 = idx_b * S2

        topk_base = idx_t1 * stride_it1
        q_base = idx_t1 * stride_qt1
        wk_base = pid * stride_wkt1
        wdq_base = pid * stride_wqkt1
        wdqt_base = pid * stride_wscoret1
        wsv_base = pid * stride_wsvt1
        wsacc_base = idx_t1 * stride_ot1

        sub_vec_id = tle.sub_vec_id()
        sub_vec_num = tle.sub_vec_num()
        CACHE_BLOCK: tl.constexpr = 64
        offs_sub_Dv = tl.arange(0, D_qk)
        sliced_row_num = BLOCK_SBS // sub_vec_num
        row_offset = sliced_row_num * sub_vec_id

        HEAD_LOOP_TIMES: tl.constexpr = triton.cdiv(G, BLOCK_G)
        BLOCK_SUB_G: tl.constexpr = BLOCK_G // 2
        sbs_loop_times = tl.cdiv(K, BLOCK_SBS)

        for idx_n1_blk in range(0, HEAD_LOOP_TIMES):
            offs_G = idx_n1_blk * BLOCK_G + tl.arange(0, BLOCK_G)
            arange_sub_G = sub_vec_id * BLOCK_SUB_G + tl.arange(0, BLOCK_SUB_G)
            offs_sub_G = idx_n1_blk * BLOCK_G + arange_sub_G
            lse = tl.zeros([BLOCK_SUB_G], dtype=tl.float32) - float("inf")

            # 1. 用 SIMD 索引 gather 稀疏 KV 到 workspace
            for idx_topk_block in range(0, sliced_row_num, CACHE_BLOCK):
                idx_offset = row_offset + idx_topk_block + tl.arange(0, CACHE_BLOCK)
                sparse_ids = tl.load(Indices_ptr + topk_base + idx_offset * stride_ik) + beg_s2
                result = tle.index_select_simd(
                    K_ptr,
                    dim=0,
                    index=sparse_ids,
                    src_shape=[cur_s2 * B, D_qk],
                    src_offset=[-1, 0],
                    read_shape=[-1, D_qk],
                )
                tl.store(Wksp_K_ptr + wk_base + idx_offset[:, None] * stride_wksbs +
                         offs_sub_Dv[None, :] * stride_wkd, result)

            # 2. Cube: q @ k^T
            for idx_sub_sbs in range(0, sbs_loop_times):
                offs_sbs = idx_sub_sbs * BLOCK_SBS + tl.arange(0, BLOCK_SBS)
                qk_l1 = cube1(Q_ptr, Wksp_K_ptr, ...)
                tl.store(Wksp_qk_ptr + wdq_base + ...)

                # 3. Vector: online softmax
                qk = tl.load(...)
                qk *= scale
                mask = offs_sbs < K
                qk = tl.where(mask[None, :], qk, -float("inf"))
                local_max = tl.max(qk, 1)
                local_exp = tl.exp(qk - local_max[:, None])
                local_sum = tl.sum(local_exp, 1)
                local_lse = local_max + tl.log(local_sum)
                # ... 更新 running lse ...

                # 4. Cube: p @ v
                # ...

            tl.store(O_ptr + wsacc_base + ..., acc.to(O_ptr.dtype.element_ty))
```

关键实现细节：
- `tle.index_select_simd` 按 SIMD 方式 gather 稀疏 KV，避免标量索引；
- `cube1` 调用 Ascend Cube 单元执行矩阵乘；
- online softmax 保持数值稳定，中间累加用 fp32；
- workspace 多级 tiling：K 缓存、qk score、sv 累加分层管理，适配 SRAM 容量。

## 已知实例

- `exp_attn_sda_001` — Hardware Adaptation（收益 35%）
  - Split-decode attention：适配 Ascend HIVM、Cube loops、多缓冲流水、混合精度累加。
- `exp_attn_bwd_q_down_001` — Hardware Adaptation（收益 35%）
  - SDA backward Q down kernel：HiAI VM 自动均衡、多缓冲、序列分片，与 up kernel 配合覆盖完整序列。
- `exp_gqa_paged_npu_001` — Hardware Adaptation（收益 45%）
  - GQA paged attention：利用 `bind_sub_block` 发挥双 Vector 核心并行，Cube/Vector 流水，按 page 迭代 KV cache。
- `exp_sparse_flash_attn_prefill_001` — Hardware Adaptation（收益 60%）
  - 稀疏 Flash Attention prefill：稀疏索引 SIMD gather、`cube1` 加速、online softmax、workspace tiling，代表 kernel 见上。
- `exp_flashattn_bwd_kv_ascend_001` — Hardware Adaptation（收益 30%）
  - Flash Attention backward KV：硬件流水线均衡、多缓冲、非对称 tiling、序列分片。

## 关联知识

- 开发指南: [Attention 算子优化](../../reference/op-class-attention.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](../../ops/nsa-compress-grad.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, attention, sparse_attention
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 hivm
- 相关主题: [Ascend 硬件资源适配规则聚合](traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Matmul 硬件适配优化聚合](traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend, hardware_adaptation
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation, hivm
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation

<!-- okf:related:end -->
