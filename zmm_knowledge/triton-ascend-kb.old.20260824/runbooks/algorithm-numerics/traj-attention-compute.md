---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: attention
dimension: "3.2"
title: Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度
description: 针对 Ascend 910B 系列上 Attention 前向/反向计算路径的优化经验，核心是通过 Online Softmax 避免物化注意力矩阵、Split-K 并行化长序列、Sparse/Top-K 跳过无效计算，以及混合精度计算平衡吞吐与数值稳定。
tags: [attention, compute, online_softmax, split_k, sparse_attention, mixed_precision, gqa, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/attention/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度

## 概述

Attention 的计算路径（`QK^T → Softmax → PV`）在 Ascend 910B 上面临两个核心挑战：

1. **`S×S` 注意力矩阵的内存与计算开销**：标准实现需要物化并存储注意力分数矩阵，长序列下内存复杂度为 `O(S²)`；
2. **Cube/Vector 单元的负载不均衡与精度约束**：矩阵乘适合 Cube，softmax 规约适合 Vector，两者需要协同；同时 FP16/BF16 的指数、累加容易溢出或下溢。

本组 11 条 experience 聚焦 **Attention 的计算优化**（而非内存访问），覆盖：

- **Online Softmax**：前向、反向、Stage2 归约中维护 `m_i`/`l_i`，避免物化注意力矩阵；
- **Split-K Attention**：把 KV 序列维度切分到多个 program 并行，再用 Stage2 做 online softmax 规约；
- **Sparse / Top-K Attention**：只计算 `real_topk` 个 KV 块，跳过全序列；
- **混合精度**：`tl.dot` 用 FP16/BF16/FP8，累加和 softmax 统计用 FP32；
- **Ascend 特化**：`tile_cube_loop` hint、`cube1` intrinsic、`tle.index_select_simd` 等；
- **GQA/MQA head 复用**：一个 KV head 服务多个 Q head；
- **Attention Sinks + Sliding Window**：在 Online Softmax 中融入 sink score 和窗口裁剪。

**核心 trade-off**：

| 优化手段 | 收益 | 代价/约束 |
|---|---|---|
| Online Softmax | 内存 `O(S²)→O(S)`，减少 HBM 往返 | 需维护 FP32 `m_i`/`l_i`，增加寄存器压力 |
| Split-K | 长序列下提升 occupancy，降低单 program 内存 footprint | 需要 Stage2 归约 kernel，增加同步和 workspace |
| Sparse/Top-K | 把 `O(S²)` 降到 `O(S·TopK)` | 依赖预计算的 topk_idx，稀疏模式需稳定 |
| 混合精度 | 提升 Cube 吞吐 | softmax/gradient 统计必须用 FP32 |
| `tile_cube_loop` / `cube1` | 指导编译器把核心循环映射到 Cube | 依赖 CANN/Triton 后端支持 |
| GQA/MQA head tiling | KV 读取量按分组比例下降 | head 分组比例过大可能损失模型能力 |

## 共同模式

### 1. Online Softmax：计算路径的核心优化

本组几乎所有 experience 都依赖 Online Softmax。其核心是在遍历 KV 块时维护运行最大值 `m_i` 和指数和 `l_i`，逐步更新输出 accumulator `acc`：

```
m_ij = max(m_i, max(qk, axis=1))
alpha = exp(m_i - m_ij)
l_i = l_i * alpha + sum(exp(qk - m_ij), axis=1)
acc = acc * alpha[:, None] + exp(qk - m_ij)[:, None] * v
```

**根因**：标准 Attention 需要先算完整 `QK^T`，再做 softmax 和 PV。Online Softmax 把 softmax 的两趟扫描改为单趟，且不需要把 `S×S` 矩阵写 HBM。

**在计算优化中的特殊考量**：
- **FP32 统计量**：`m_i`、`l_i`、`acc` 必须用 FP32，否则长序列下指数和会累积显著误差。
- **精度敏感的 backward**：`dS = p * (dP - d)` 中 `p`、`dP`、`d` 的精度直接影响梯度稳定性，通常用 FP32 算 `dS` 再 cast 到 FP16/BF16 做 `tl.dot`。
- **Stage2 归约**：Split-K 产生多个 partial `(acc, m_i, l_i)`，Stage2 需要用 Online Softmax 公式把它们合并成最终结果，而不是简单求和。

**收益明显条件**：
- 序列长度 `S≥512`，越长收益越大；
- batch×heads 足够多，能隐藏 softmax 规约的延迟；
- head_dim≥64，保证每个 Q/K/V block 有足够计算量。

### 2. Split-K：长序列的 K 维并行

当 KV 序列很长时，单个 program 处理全部 KV 块会导致：
- 每个 program 的 accumulator/统计量生命周期过长，寄存器压力大；
- 可并行度受限于 `batch × heads × blocks_m`。

Split-K 把 KV 序列切成 `SPLIT_K` 份，由不同 program 并行计算 partial attention，再用 Stage2 kernel 按 Online Softmax 合并。

**根因**：Attention 的规约维度是 K（序列），天然可切分。切分后每个 partial 独立做 Online Softmax，最后合并时只需要传递 `(acc, m_i, l_i)` 三元组。

**实现要点**：
- Stage1 每个 split 输出 `partial_acc`、`partial_m`、`partial_l`；
- Stage2 按 `kv_indptr` 确定每个 split 的有效长度，用 `tl.minimum` 处理尾块；
- Sink token 的贡献在 Stage2 最终归一化时加入（`e_sum += exp(cur_sink - e_max)`）。

**与其他优化的关系**：
- Split-K 与 Online Softmax **必须同时使用**，否则 partial 结果无法正确合并。
- Split-K 与 Paged KV Cache **互补**：分页解决非连续存储，Split-K 解决长序列并行。

### 3. Sparse / Top-K Attention：跳过无效计算

对于长上下文或稀疏注意力，只有 top-K 个 KV 块对当前 Query 重要。`exp_sparse_attn_topk_001` 和 `exp_sparse_flash_attn_decode_001` 只遍历 `real_topk` 块：

**根因**：标准 Attention 的 `O(S²)` 在长序列下不可接受。若注意力模式本身稀疏，可直接跳过零值或低分 KV 块。

**实现要点**：
- 用 `t_ptr`（topk_idx）拿到当前 Q 块需要访问的 KV 块索引；
- 用 `tle.index_select_simd` gather 对应的 K/V；
- causal mask 直接在计算循环内用 `tl.where` 应用；
- Online Softmax 仍然适用，因为遍历的块数减少但公式不变。

**收益明显条件**：
- TopK ≪ S；
- 稀疏模式稳定（如局部注意力、block sparse、Top-K routing）；
- 有预计算 topk_idx 的开销可被节省的计算覆盖。

**常见坑**：
- gather 访问不规则，需确保 memory coalescing 或使用 SIMD gather；
- 变长序列用 `cu_seqlens` 定位每个样本的 topk 范围；
- GQA 场景下 `NUM_SHARE_Q_HEADS` 要正确计算 KV head 共享。

### 4. 混合精度：Cube 吞吐 + FP32 数值稳定

本组 backward 经验（`exp_flashattn_bwd_kv_001`、`exp_attn_bwd_q_001`、`exp_sdpa_bwd_q_001`）都强调混合精度：

- **矩阵乘**（`tl.dot`）用低精度输入（FP16/BF16/FP8）以利用 Cube 高吞吐；
- **softmax 统计、梯度、累加**用 FP32；
- **最终输出** cast 回低精度存储。

**根因**：FP16/BF16 的指数和累加容易溢出/下溢，但 Cube 在 FP16/BF16 下吞吐远高于 FP32。把“数值敏感但计算量小”的部分放在 FP32，“计算量大但容错”的部分放在低精度，是最佳平衡。

**实现要点**：
- 前向中 `p = exp(qk - m_ij)` cast 到 `k.dtype` 后再与 V 做 dot；
- 反向中 `dS` 用 FP32 计算，再 cast 到 LOW_TYPE 做 `dQ = dS @ K`、`dK = dS^T @ Q`；
- 对 FP8 Value，需要 UB slicing 防止溢出（参考 attention Memory Access 卡）。

### 5. Ascend 特化编译 Hint 与 Intrinsics

- **`tl.compile_hint(acc_ptr, "tile_cube_loop", 2)`**：告诉编译器把 accumulation loop 映射到 Cube unit，提升矩阵乘吞吐。
- **`cube1` intrinsic**：用于稀疏 Attention 中的 QK dot，直接调用 Ascend Cube 指令。
- **`tle.index_select_simd`**：SIMD gather，从不规则索引中并行取 K/V。
- **`tle.sub_vec_id`**：把向量加载分布到不同 vector lane。

**根因**：Triton 通用 IR 不一定能自动把循环映射到 Cube 或选择最优 gather 指令。显式 hint/intrinsic 可以弥补编译器启发式的不足。

### 6. GQA/MQA Head Tiling 与 RoPE 融合

- **GQA**：`BLOCK_H` 个 query head 共享一个 KV head，一次加载 KV 服务多个 Q head。
- **MQA RoPE**：`exp_rope_mqa_001` 把 Q 和 K 的 RoPE 融合到一个 kernel，用 `tl.constexpr` 区分 Neox/GPT-J style 和非旋转维度 pass-through。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_sdpa_infer_ascend_001` | 50% | SDPA 推理内层循环，Online Softmax + `tile_cube_loop` hint + UB-safe `tl.where` masking + FP8 支持，代表 kernel。 |
| `exp_sparse_flash_attn_decode_001` | 45% | 稀疏 FlashAttention decode，`tle.index_select_simd` gather Top-K KV + `cube1` QK dot + L0/L1 workspace tiling。 |
| `exp_sparse_attn_topk_001` | 40% | Top-K sparse attention 前向，只遍历 `real_topk` 块，Online Softmax + causal mask 向量化。 |
| `exp_flashattn_bwd_kv_001` | 35% | FlashAttention backward 中 dK/dV 微内核，混合精度（FP16/BF16 dot + FP32 softmax gradient）+ in-place 累加。 |
| `exp_attn_bwd_q_001` | 35% | Attention backward 中 dQ 微内核，序列方向 tiling + 混合精度 + gradient 寄存器累加。 |
| `exp_flashattn_splitk_gqa_001` | 35% | Split-K GQA attention Stage1，K 维切分 + Online Softmax + paged KV + GQA head tiling + logit softcapping。 |
| `exp_attn_sinks_prefill_001` | 35% | Attention Sinks prefill，Br=16 PAGE_SIZE=16，sink score 初始化 Online Softmax + sliding window 裁剪 + paged KV。 |
| `exp_attn_reduce_stage2_001` | 30% | Split-K Stage2 归约，跨 KV splits 做 Online Softmax reduction，FP32 累加，sink token 最终归一化。 |
| `exp_rope_mqa_001` | 30% | MQA RoPE，Q/K 融合 + 向量化加载 + Neox/GPT-J constexpr + 非旋转维度 pass-through。 |
| `exp_sdpa_bwd_q_001` | 30% | SDPA backward dQ，autotune BLOCK_R/C + 混合精度 + multibuffer prefetching。 |
| `exp_attn_qk_cube_001` | 25% | QK^T batched matmul，D_qk 方向 tiling + `tl.dot` 走 Cube + FP32 累加 + 边界 mask。 |

## 代表 Kernel

选取 `exp_sdpa_infer_ascend_001`（`216-_sdpa_infer_inner.py`）作为代表，因为它同时体现了 Attention 计算优化中最关键的三个技术：

1. **Online Softmax**：迭代更新 `m_i`、`l_i`、`acc_ptr`；
2. **Ascend Cube 特化**：`tl.compile_hint(acc_ptr, "tile_cube_loop", 2)`；
3. **混合精度与 UB 安全**：`p_cast = p.to(k.dtype)` 后做 dot，用 `tl.where` 替代 `-inf` 避免 UB 溢出。

```python
@triton.jit
def _sdpa_infer_inner(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask_base_ptr,
    start_m,
    qk_scale,  # Starting position of current query block, qk scale factor
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Block size constants
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  # Current stage flag, m and n offset indices
    SEQ: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # Iterate over all k, v blocks in the current stage and accumulate the output
    for start_n in range(0, SEQ, BLOCK_N):  # Process BLOCK_N columns at a time
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
        mask_ptr = (
            mask_base_ptr
            + start_m * BLOCK_M * SEQ
            + start_n
            + tl.arange(0, BLOCK_M)[:, None] * SEQ
            + tl.arange(0, BLOCK_N)[None, :]
        )
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)

        # NOTE(zhangjihang): tl.where will introduce ub overflow
        qk = qk * qk_scale
        mask = tl.load(mask_ptr)

        # qk += (1 - mask.to(tl.float32)) * (-1e6)
        # qk = tl.where(mask, qk, float("-inf"))
        qk = tl.where(mask, qk, -1e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
        qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr)  # Load corresponding V block
        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        acc_ptr = acc_ptr * alpha[:, None]
        acc_ptr = tl.dot(p_cast, v, acc_ptr)
        tl.compile_hint(acc_ptr, "tile_cube_loop", 2)

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i
```

## 已知实例

- `exp_sparse_flash_attn_decode_001` — Compute （收益 45%）
  - 稀疏 FlashAttention decode，`tle.index_select_simd` gather Top-K KV + `cube1` QK dot + L0/L1 workspace tiling。
  - 对应 kernel: `raw_experiences/extracted_kernels/196-sparse_flash_attention_decode_kernel.py`
- `exp_flashattn_bwd_kv_001` — Compute （收益 35%）
  - FlashAttention backward 中 dK/dV 微内核，混合精度 + in-place 累加。
  - 对应 kernel: `raw_experiences/extracted_kernels/258-micro_kernel_bwd_kv.py`
- `exp_sparse_attn_topk_001` — Compute （收益 40%）
  - Top-K sparse attention 前向，只遍历 `real_topk` 块，Online Softmax + causal mask 向量化。
  - 对应 kernel: `raw_experiences/extracted_kernels/020-forward_kernel.py`
- `exp_attn_bwd_q_001` — Compute （收益 35%）
  - Attention backward 中 dQ 微内核，序列方向 tiling + 混合精度 + gradient 寄存器累加。
  - 对应 kernel: `raw_experiences/extracted_kernels/257-micro_kernel_bwd_q.py`
- `exp_flashattn_splitk_gqa_001` — Compute （收益 35%）
  - Split-K GQA attention Stage1，K 维切分 + Online Softmax + paged KV + GQA head tiling + logit softcapping。
  - 对应 kernel: `raw_experiences/extracted_kernels/017-_fwd_grouped_split_kernel.py`
- `exp_attn_reduce_stage2_001` — Compute （收益 30%）
  - Split-K Stage2 归约，跨 KV splits 做 Online Softmax reduction，sink token 最终归一化。
  - 对应 kernel: `raw_experiences/extracted_kernels/028-_fwd_kernel_stage2.py`
- `exp_sdpa_infer_ascend_001` — Compute （收益 50%）
  - SDPA 推理内层循环，Online Softmax + `tile_cube_loop` hint + UB-safe masking，代表 kernel。
  - 对应 kernel: `raw_experiences/extracted_kernels/216-_sdpa_infer_inner.py`
- `exp_rope_mqa_001` — Compute （收益 30%）
  - MQA RoPE，Q/K 融合 + 向量化加载 + Neox/GPT-J constexpr + 非旋转维度 pass-through。
  - 对应 kernel: `raw_experiences/extracted_kernels/192-fused_rope_qk_mqa_kernel.py`
- `exp_sdpa_bwd_q_001` — Compute （收益 30%）
  - SDPA backward dQ，autotune + 混合精度 + multibuffer prefetching。
  - 对应 kernel: `raw_experiences/extracted_kernels/kernel_sdpa_bwd_q`
- `exp_attn_sinks_prefill_001` — Compute （收益 35%）
  - Attention Sinks prefill，Br=16 PAGE_SIZE=16，sink score 初始化 Online Softmax + sliding window 裁剪 + paged KV。
  - 对应 kernel: `raw_experiences/extracted_kernels/174-attention_sinks_prefill_kernel.py`
- `exp_attn_qk_cube_001` — Compute （收益 25%）
  - QK^T batched matmul，D_qk 方向 tiling + `tl.dot` 走 Cube + FP32 累加。
  - 对应 kernel: `raw_experiences/extracted_kernels/198-cube1.py`

## 关联知识

- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, attention, gqa
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, split_k
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention, gqa, sparse_attention
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention, gqa, online_softmax, sparse_attention
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention, gqa, sparse_attention
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention, gqa
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](../../ops/nsa-compress-grad.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../../ops/nsa-selected-attention.md) — 主题相关：共享 attention, gqa, online_softmax
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 attention
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention, online_softmax, sparse_attention
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention, online_softmax, sparse_attention
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](../compute-pipeline/traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D Compute 优化聚合](../compute-pipeline/traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](../compute-pipeline/traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](../compute-pipeline/traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute, online_softmax
- 相关主题: [Matmul Compute 优化聚合](../compute-pipeline/traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../compute-pipeline/traj-matmul-split.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](../compute-pipeline/traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute, split_k
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](../compute-pipeline/traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute, online_softmax
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](../compute-pipeline/traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](../compute-pipeline/traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, attention
- 相关主题: [Attention 硬件适配优化聚合](../portability/traj-attention-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, attention, sparse_attention
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 主题相关：共享 mixed_precision
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 主题相关：共享 online_softmax
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttention 变体的内存访问优化](traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, attention, gqa, online_softmax
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3, attention, online_softmax
- 相关主题: [Normalization Compute 优化聚合](traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
