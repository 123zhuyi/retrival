---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: attention
dimension: "3.2"
title: FlashAttention 变体的内存访问优化
description: 针对 Ascend 910B 系列上 FlashAttention 及其变体（GQA/MQA、Paged Attention、稀疏/滑动窗口 Attention）的内存访问优化经验，核心是通过 Online Softmax、Block Pointer 与 KV Cache 分页将 Attention 的内存复杂度从 O(S²) 降到 O(S)。
tags: [attention, flash_attention, memory_access, online_softmax, block_pointer, paged_attention, gqa, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/attention/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# FlashAttention 变体的内存访问优化

## 概述

Attention 的标准实现需要在 HBM 中物化 `S×S` 的注意力分数矩阵（`S` 为序列长度），导致内存复杂度为 `O(S²)`。当序列较长（`S≥1024`）或 batch 较大时，HBM 读写成为瓶颈，甚至超过计算开销。

本组 22 条 experience 均围绕 **FlashAttention 系列变体** 在 Ascend 910B2/910B3 上的内存访问优化，覆盖以下场景：

- **标准 FlashAttention 前向 / SDPA 推理**：通过 Online Softmax 避免物化注意力矩阵。
- **GQA/MQA**：多个 query head 共享同一组 KV head，减少 KV Cache 读取带宽。
- **Paged Attention / Paged KV Cache**：用 block table 映射逻辑页到非连续物理页，支持变长序列和动态显存分配。
- **稀疏 / Top-K / 滑动窗口 Attention**：只加载必要的 K/V 块，跳过零值或窗口外计算。
- **HSTU / Lightning Attention 等特殊结构**：在单一 kernel 内融合 QK^T、激活、归一化与 PV，避免中间结果写回 HBM。

**核心 trade-off**：

| 优化手段 | 收益 | 代价/约束 |
|---|---|---|
| Online Softmax | 内存 `O(S²)→O(S)`，通常提升 30%–200% | 需维护 `m_i`/`l_i` 的 FP32 运行统计量，增加寄存器压力 |
| Block Pointer (`tl.make_block_ptr`) | 向量化的 2D 分块加载，提升带宽利用率 | 对非连续/变长布局需要显式处理 stride 和 boundary check |
| Paged KV Cache | 支持变长、动态 batch，减少显存碎片 | 引入 block table 间接寻址，可能降低访问规律性 |
| GQA/MQA head 共享 | KV Cache 带宽与容量按 `num_kv_heads/num_q_heads` 比例下降 | 当分组比例过大时可能损失表达能力 |
| FP8 + UB slicing | 降低 PV 阶段数据搬运量 | 仅在大 head_dim（≥256）时需要切片避免 UB 溢出 |
| Kernel Fusion | 避免中间矩阵写 HBM | kernel 体积增大，寄存器/共享内存预算受限 |

**适用条件**：`S≥128`、head_dim≥64、fp16/bf16，典型收益在 30%–200% 之间，长序列和 GQA 场景收益更高。

## 共同模式

### 1. Online Softmax：Attention 内存优化的核心

几乎所有本组 experience 都将 Online Softmax 作为首要优化点。其本质是把标准 Softmax 的两趟扫描改为单趟迭代：在处理每个 K/V 块时，维护当前最大值 `m_i` 和指数和 `l_i`，并据此缩放已积累的输出 `acc`。

```
m_ij = max(m_i, max(Q_i @ K_j^T / sqrt(d_k)))
alpha = exp(m_i - m_ij)
l_i   = l_i * alpha + sum(exp(Q_i @ K_j^T / sqrt(d_k) - m_ij))
acc   = acc * alpha + exp(Q_i @ K_j^T / sqrt(d_k) - m_ij) @ V_j
```

**根因**：标准 Attention 需要先把 `S×S` 的 `QK^T` 写 HBM，再做 Softmax 和 PV，HBM 读写量与 `S²` 成正比。Online Softmax 让 `acc`、`m_i`、`l_i` 始终驻留寄存器/SRAM，只将最终 `O` 写回 HBM。

**收益明显条件**：
- 序列长度 `S≥512`，越长收益越大；
- head_dim 较大（≥64），此时 Q/K/V 块本身就有足够计算量来隐藏内存延迟；
- batch×heads 数量足够多，能把 core 跑满。

**实现要点与常见坑**：
- `m_i`、`l_i`、`acc` 必须用 **FP32** 累加，否则指数和规约会累积数值误差。
- 每次迭代要先算新的 `m_ij`，再缩放旧的 `acc` 和 `l_i`，顺序不能颠倒。
- 处理 causal mask 时，把 mask 直接合并到 `qk` 上（`qk = qk * scale + where(mask, -inf, 0)`），避免分支。
- 滑动窗口 / 稀疏 Attention 需要先确定当前 Q 块实际要访问的 K/V 范围，避免加载无效块。

### 2. Block Pointer：让 2D 分块访问向量化

`tl.make_block_ptr` 被反复用于 Q/K/V/O 的加载与存储。相比手动指针计算，Block Pointer 把 strides、offsets、block_shape、order 显式告诉编译器，使其能生成对齐的向量化 load/store，并自动处理边界填充。

**根因**：Attention 中 Q/K/V 通常是 4D 张量（batch, head, seq, dim），手动指针计算容易破坏合并访问；Block Pointer 抽象后编译器可以一次性发出 2D block 传输指令。

**收益明显条件**：
- 张量布局规则（如 `[batch, head, seq, dim]`）且 stride 已知；
- BLOCK_M/BLOCK_N 是硬件向量宽度（32/64 byte）的整数倍。

**常见坑**：
- 对 paged KV cache 等间接索引场景，Block Pointer 的 base 地址需要动态计算（`base + page_offset`），不要直接对 block table 做 Block Pointer。
- `boundary_check` 和 `padding_option="zero"` 要配对使用，否则尾块可能越界。
- `order` 参数必须与 tensor 的内存布局一致，否则生成的访问模式不连续。

### 3. Paged KV Cache：变长序列的内存管理

LLM 推理中每个请求的序列长度不同，且随 decode 不断增长。固定形状的 KV Cache 会造成显存碎片和浪费。Paged Attention 把 KV 切分为固定大小的页，通过 `block_table` 把逻辑块映射到物理块。

**根因**：非连续物理存储避免了为每个请求预留最大长度；动态分配页可以复用空闲块。

**收益明显条件**：
- batch 内序列长度差异大；
- decode 阶段上下文持续追加；
- 使用 GQA/MQA 时，KV Cache 容量本身较小，分页带来的管理收益更明显。

**实现要点**：
- 计算 K/V 物理地址时：`page_id = block_table[batch, logical_block]；base = k_cache_ptr + page_id * page_size * stride`。
- 对每个 Q 块，先根据 `kv_len` 确定需要访问的逻辑页范围，再逐页加载。
- 页大小通常取 BLOCK_N 的整数倍，Prefill 场景要求 `PAGE_SIZE == BLOCK_SIZE_N`。

### 4. GQA/MQA：用 head 共享降低 KV 带宽

GQA 让多个 query head 共享同一个 KV head，MLA 则进一步把 KV 压缩到 latent 维度。两者都减少了需要加载的 KV 数据量。

**根因**：Attention 的内存带宽主要消耗在 K/V 加载上（每个 Q 块要遍历所有 K/V 块）。`num_kv_heads << num_q_heads` 时，KV 读取量成比例下降。

**实现要点**：
- 在 kernel 内通过 `kv_head_id = q_head_id // q_heads_per_kv_head` 计算共享关系，避免重复加载。
- 对于 GQA，可在单个 program 中同时处理 `BLOCK_H` 个 query head，把一次 K/V 加载复用到多个 Q head 上。
- MLA 需要把 Key 拆分为 content（NOPE）和 positional（ROPE）两部分，Value 与 NOPE 共享 buffer，注意 stride 和 layout 匹配。

### 5. Kernel Fusion：把多算子合并为单一 kernel

HSTU、Lightning Attention 等 experience 强调把 `QK^T → activation → Softmax → PV` 融合在一个 kernel 里。融合后中间矩阵（S、P）不经过 HBM，显著降低带宽。

**与其他优化的关系**：
- 融合与 Online Softmax **互补**：Online Softmax 解决了内存物化问题，融合进一步减少了 launch 开销和中间张量写入。
- 融合与 Paged KV Cache **互补**：分页解决变长存储，融合解决中间计算带宽。
- 融合可能增加寄存器压力，当与 FP8 UB slicing 或大 head_dim 同时使用时，需要留出足够 SRAM 预算。

### 6. 稀疏 / 滑动窗口 / Top-K Attention：减少无效加载

稀疏 Attention 只加载 `topk_q_idx` 指定的 K/V 块；滑动窗口只加载窗口内的块；causal mask 天然跳过上三角。

**实现要点**：
- 在循环前根据 `cu_seqlens`、`window_size` 或 `topk_idx` 确定有效 K/V 范围，避免无效 `tl.load`。
- 对稀疏 backward，把 `do`、`lse`、`d` 等 per-Q-token 数据提升到内层循环外加载一次，再复用给所有 Top-K K 块。
- 边界处理用 `tl.where` 做向量化 mask，避免分支发散。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_flash_attn_001` | 200% | FlashAttention 前向，支持变长序列（`Start_q`）、滑动窗口（`BANDWIDTH`）与 Attention Sinks，使用 Online Softmax + Block Pointer。 |
| `exp_sdpa_flash_001` | 200% | FlashAttention 内层循环，融合 QK^T、Online Softmax、PV，支持 FP8 Value 与 `multibuffer` 软件流水。 |
| `exp_flashattn_fwd_001` | 200% | FlashAttention 前向微内核，预加载 `block_q` 后沿 KV 维度迭代更新输出与 Softmax 统计量。 |
| `exp_sdpa_infer_001` | 150% | SDPA 推理 kernel，省略 backward 与 logsumexp 存储，支持 FP8 Value，使用 grid-stride 调度与 GQA。 |
| `exp_flashattn_gqa_001` | 45% | GQA/MQA FlashAttention Stage1，使用 `libdevice.index_select_simd` 处理不规则索引，Split-KV 并行。 |
| `exp_mla_paged_attn_001` | 45% | MLA 分页 Attention，KV 压缩为 latent 维度，Value 与 Key content 共享 buffer，减少 KV Cache 容量。 |
| `exp_attn_sparse_bwd_001` | 40% | 稀疏 Attention backward（Top-K/Local），通过 `topk_q_idx` gather 只加载相关 Q/DO/LSE/D，K/V tile 复用。 |
| `exp_attn_sinks_001` | 40% | 带 Attention Sinks 的 decode kernel，预计算 sink score 并融入 Online Softmax，支持 paged KV cache 与滑动窗口。 |
| `exp_paged_kv_cache_001` | 40% | Paged KV Cache 写入优化，将 head 维度向量化，把串行 head 循环改为宽 vectorized store。 |
| `exp_sparse_attn_backward_dq_001` | 40% | 稀疏 backward 中 dQ 计算，基于 `cu_seqlens` 处理变长，持久化复用 `do`/`lse`/`d`，使用 `tl.exp2` 加速激活。 |
| `exp_paged_attn_flash_001` | 35% | FlashAttention 风格 Paged Attention，支持 GQA、变长序列、滑动窗口与 logit softcapping。 |
| `exp_hstu_attn_fwd_001` | 35% | HSTU Attention 前向，融合 SiLU 激活、归一化与 PV，使用 Block Pointer 处理变长 `seq_offsets`。 |
| `exp_paged_prefill_001` | 35% | Paged Attention Prefill，要求 `PAGE_SIZE == BLOCK_SIZE_N`，核心计算委托给 `_sdpa_infer_single_block`。 |
| `exp_grouped_attn_stage1_001` | 35% | Grouped Attention Stage1，对 K 使用 scalar gather（`tl.get_element`/`tl.insert_slice`）解决不规则索引，V 保持向量化加载。 |
| `exp_paged_gqa_001` | 35% | Paged GQA decode，单 program 处理多个 query head 分摊 K/V 加载，并重叠 V 加载与 QK 点乘。 |
| `exp_attn_bwd_hstu_001` | 35% | HSTU Attention backward，按 K/V column block 外提加载并在多 row block 间复用，DK/DV 用 FP32 累加。 |
| `exp_paged_attn_decode_001` | 30% | Paged Attention decode，context length 在 batch 内变化，通过 block table 访问非连续 KV。 |
| `exp_attn_fp8_ub_slice_001` | 30% | FP8 FlashAttention，将 softmax 权重 cast 到 FP8 后再做 PV，HEAD_DIM≥256 时对 accumulator 切片避免 UB 溢出。 |
| `exp_lightning_attn_decode_001` | 30% | Lightning Attention decode，将 q/k/v 加载提升到 `pid_d` 循环外，消除冗余内存访问。 |
| `exp_attn_causal_mask_001` | 15% | Causal mask 辅助函数，用 `tl.arange` 广播生成偏移、min/max 边界夹紧，支持滑动窗口。 |
| `exp_sdpa_bwd_reduce_001` | 25% | SDPA backward 中 `d` 的 reduction kernel，使用 `multibuffer` 软件流水、FP32 累加与向量化 mask load。 |
| `exp_attn_ptr_setup_001` | 25% | FlashAttention 指针准备工具，将 BLOCK_M 4 等分为 SUB_BLOCK_M，预计算 workspace 指针以隐藏地址计算延迟。 |

## 代表 Kernel

选取 `exp_flash_attn_001`（`088-_attn_fwd.py`）作为代表，因为它同时覆盖了本组最核心的三个模式：**Block Pointer 2D 加载**、**Online Softmax 迭代**、**Causal/滑动窗口/Attention Sinks mask 融合**。其结构与大多数 FlashAttention 变体同构，便于迁移到其他场景（GQA、Paged、FP8 等）。

```python
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    Start_q,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,  #
    Z,
    H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_Q_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_KV_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_KV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_Q_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = tl.load(Q_block_ptr)

    if BANDWIDTH:
        lo, hi = tl.maximum(start_q, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = start_q, start_q + (start_m + 1) * BLOCK_M

    # advance the KV block-pointers so they point at `lo`
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr).to(tl.float32)

        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    m_i += tl.math.log(l_i)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
```

## 已知实例

- `exp_sdpa_infer_001` — Memory Access （收益 150%）
  - 面向推理的 SDPA kernel，省略 backward 逻辑并将 logsumexp 保留在寄存器；支持 FP8 Value 与 GQA。
  - 对应 kernel: `raw_experiences/extracted_kernels/217-_sdpa_infer_kernel.py`
- `exp_paged_prefill_001` — Memory Access （收益 35%）
  - Paged Attention 的 prefill 阶段实现，要求 `PAGE_SIZE == BLOCK_SIZE_N`，使用 block table 管理非连续 KV。
  - 对应 kernel: `raw_experiences/extracted_kernels/paged_prefill_kernel.py`
- `exp_flash_attn_001` — Memory Access （收益 200%）
  - FlashAttention 前向，支持变长序列（`Start_q`）、滑动窗口（`BANDWIDTH`）与 Attention Sinks，核心代表 kernel。
  - 对应 kernel: `raw_experiences/extracted_kernels/088-_attn_fwd.py`
- `exp_paged_attn_flash_001` — Memory Access （收益 35%）
  - 支持 GQA、变长序列（`Q_seqlens`/`KV_seqlens`）、滑动窗口与 logit softcapping 的 Paged FlashAttention。
  - 对应 kernel: `raw_experiences/extracted_kernels/019-_fwd_kernel.py`
- `exp_hstu_attn_fwd_001` — Memory Access （收益 35%）
  - HSTU Attention 前向，融合 SiLU、归一化与 PV；用 Block Pointer 处理变长 `seq_offsets`。
  - 对应 kernel: `raw_experiences/extracted_kernels/006-_hstu_attn_fwd_compute.py`
- `exp_attn_fp8_ub_slice_001` — Memory Access （收益 30%）
  - FP8 FlashAttention，PV 前将 softmax 权重 cast 到 FP8；HEAD_DIM≥256 时对 accumulator 切片避免 UB 溢出。
  - 对应 kernel: `raw_experiences/extracted_kernels/011-_attn_fwd_inner.py`
- `exp_attn_causal_mask_001` — Memory Access （收益 15%）
  - Causal mask 辅助函数，使用 `tl.arange` 广播与 min/max 夹紧生成偏移，支持滑动窗口。
  - 对应 kernel: `raw_experiences/extracted_kernels/252-causal_mask_fn.py`
- `exp_attn_sparse_bwd_001` — Memory Access （收益 40%）
  - Top-K / Local Attention 的 backward，依据 `topk_q_idx`  gather 相关 Q/DO/LSE/D，K/V tile 在 row block 间复用。
  - 对应 kernel: `raw_experiences/extracted_kernels/025-backward_dkdv.py`
- `exp_flashattn_gqa_001` — Memory Access （收益 45%）
  - GQA/MQA FlashAttention Stage1，使用 `libdevice.index_select_simd` 处理不规则索引，Split-KV 并行。
  - 对应 kernel: `raw_experiences/extracted_kernels/027-_fwd_kernel_stage1.py`
- `exp_grouped_attn_stage1_001` — Memory Access （收益 35%）
  - Grouped Attention Stage1，对 K 使用 scalar gather 解决不规则索引，V 保持向量化加载。
  - 对应 kernel: `raw_experiences/extracted_kernels/013-grouped_attention_kernel_stage1.py`
- `exp_sdpa_flash_001` — Memory Access （收益 200%）
  - FlashAttention 内层循环，融合 QK^T、Online Softmax、PV，支持 FP8 Value 与 `multibuffer` 软件流水。
  - 对应 kernel: `raw_experiences/extracted_kernels/253-_sdpa_infer_single_block.py`
- `exp_flashattn_fwd_001` — Memory Access （收益 200%）
  - FlashAttention 前向微内核，预加载 `block_q` 后沿 KV 维度迭代更新输出与 Softmax 统计量。
  - 对应 kernel: `raw_experiences/extracted_kernels/256-micro_kernel_fwd.py`
- `exp_attn_sinks_001` — Memory Access （收益 40%）
  - 带 Attention Sinks 的 decode kernel，预计算 sink score 并融入 Online Softmax。
  - 对应 kernel: `raw_experiences/extracted_kernels/173-attention_sinks_kernel.py`
- `exp_paged_attn_decode_001` — Memory Access （收益 30%）
  - LLM decode 阶段 Paged Attention，batch 内序列长度不同，使用 paged KV cache。
  - 对应 kernel: `raw_experiences/extracted_kernels/255-paged_decode_kernel.py`
- `exp_paged_kv_cache_001` — Memory Access （收益 40%）
  - Paged KV Cache 写入优化，将 head 维度向量化，把串行 head 循环改为宽 vectorized store。
  - 对应 kernel: `raw_experiences/extracted_kernels/238-_store_paged_kv_cache_kernel.py`
- `exp_attn_bwd_hstu_001` — Memory Access （收益 35%）
  - HSTU Attention backward，K/V tile 在 row block 间复用，DK/DV 用 FP32 累加。
  - 对应 kernel: `raw_experiences/extracted_kernels/009-_hstu_attn_bwd_one_col_block.py`
- `exp_sdpa_bwd_reduce_001` — Memory Access （收益 25%）
  - SDPA backward 中 `d` 的 reduction kernel，使用 `multibuffer` 软件流水、FP32 累加与向量化 mask load。
  - 对应 kernel: `raw_experiences/extracted_kernels/219-kernel_sdpa_bwd_d.py`
- `exp_lightning_attn_decode_001` — Memory Access （收益 30%）
  - Lightning Attention decode，将 q/k/v 加载提升到 `pid_d` 循环外，消除冗余内存访问。
  - 对应 kernel: `raw_experiences/extracted_kernels/069-_lightningattn_attn_decode_kernel.py`
- `exp_sparse_attn_backward_dq_001` — Memory Access （收益 40%）
  - 稀疏 backward 中 dQ 计算，基于 `cu_seqlens` 处理变长，持久化复用 `do`/`lse`/`d`。
  - 对应 kernel: `raw_experiences/extracted_kernels/026-backward_dq.py`
- `exp_paged_gqa_001` — Memory Access （收益 35%）
  - Paged GQA decode，单 program 处理多个 query head 分摊 K/V 加载。
  - 对应 kernel: `raw_experiences/extracted_kernels/176-_paged_gqa_fwd_kernel.py`
- `exp_mla_paged_attn_001` — Memory Access （收益 45%）
  - MLA 分页 Attention，Value 与 Key content（K_NOPE）共享 buffer，KV 压缩为 latent 维度。
  - 对应 kernel: `raw_experiences/extracted_kernels/_paged_mla_fwd_kernel`
- `exp_attn_ptr_setup_001` — Memory Access （收益 25%）
  - FlashAttention 指针准备工具，将 BLOCK_M 4 等分为 SUB_BLOCK_M，预计算 workspace 指针。
  - 对应 kernel: `raw_experiences/extracted_kernels/053-get_params.py`

## 关联知识

- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 910b2, 910b3, attention, flash_attention, gqa
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention, gqa
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention, flash_attention, gqa, online_softmax
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 attention, flash_attention, online_softmax, paged_attention
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 主题相关：共享 attention, flash_attention, online_softmax, paged_attention
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention, gqa
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention, gqa, paged_attention
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../../ops/nsa-selected-attention.md) — 主题相关：共享 attention, gqa, online_softmax
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention, flash_attention, online_softmax, paged_attention
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, memory_access
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../compute-pipeline/traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, block_pointer
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Activation Backward 内存访问优化聚合](../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](../memory-sram/traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, attention, gqa, online_softmax
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
