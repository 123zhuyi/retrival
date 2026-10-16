---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: Fused Attention / Flash Attention 高性能算子
description: Fused Attention / Flash Attention 高性能算子
tags: [attention, flash_attention, softmax, gqa, mqa, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/attention.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/attention.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/attention.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# Fused Attention / Flash Attention 高性能算子

## 概述

Attention 机制是 Transformer 架构的核心计算模块，标准实现需要在 HBM 中物化 S×S 的注意力分数矩阵（S 为序列长度），内存复杂度为 O(S²)。对于长序列（S≥1024），这一开销成为性能瓶颈。Flash Attention 通过 Online Softmax 算法将注意力计算分块完成，内存复杂度降至 O(S)，同时利用 Ascend Cube 单元的高带宽显著提升计算效率。

在 Ascend 910b 系列上，Fused Attention 的优化重点在于：Q/K/V 数据的分块加载（tiling）、Online Softmax 的数值稳定计算、GQA/MQA 的 head 共享、以及 prefill/decode 两种场景的差异化调度。基础实现相比 PyTorch 多算子基线提升 15%，Flash Attention 路径可达 50% 提升。

## 检测标准

满足以下任意一条，认定为 Attention 类算子：

- 计算图包含 `scaled_dot_product_attention` / `F.softmax(Q @ K.T / sqrt(d_k)) @ V` 模式
- 输入形状为 `[batch, num_heads, seq_len, head_dim]` 或 `[batch, seq_len, num_heads, head_dim]`
- einsum 表达式含 `bhid,bhjd->bhij` 或 `bhij,bhjd->bhid` 等注意力收缩模式
- 算子包含可选的 `causal_mask`、`alibi_slopes` 或 `attn_bias`

## 适用条件

| 维度 | 最小值 | 推荐值 | 说明 |
|------|--------|--------|------|
| seq_len (S) | 1 | ≥1024 | S<1024 走标准 3 步路径，S≥1024 走 Flash Attention |
| head_dim (d_k) | 16 | ≥64 | d_k < 64 收益有限 |
| num_heads (H) | 1 | ≥8 | GQA/MQA 场景 num_kv_heads 可小于 num_heads |
| dtype | — | fp16 / bf16 | FP8 需配合 UB overflow slicing |
| mask | — | causal / None | 任意 mask 均支持，causal mask 有专项优化 |
| 场景 | — | prefill / decode | 两种场景调度策略不同，推荐 dual dispatch |

## 经验路由

### 基础经验

| 场景 | 经验 ID | 提升 | 核心技术 | 适用条件 |
|------|---------|------|----------|----------|
| 标准 3 步 Attention | exp_fused_attention_001 | 15% | QK^T → 缩放 → Softmax → AV，内存 O(S²) | S≥128, d_k≥64, H≥8, fp16/bf16/fp32 |
| Flash Attention Online Softmax | exp_fused_attention_002 | 50% | Online softmax 分块，内存 O(S²)→O(S) | S≥1024, d_k≥64, H≥8, fp16/bf16 |

### 高性能 Flash Attention 变体

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| FP8 量化注意力 | exp_flash_attention_fp8_001 | 35% | FP8 + UB overflow slicing 防溢出 |
| Grid-stride + 软件流水线 | exp_flash_attention_pipeline_001 | 45% | grid_stride_loop + 软件 pipelining |
| Split-K GQA + Paged KV Cache | exp_flash_attention_splitk_001 | 40% | K 轴分裂，支持 paged KV cache |
| Paged Attention | exp_flash_attention_paged_001 | 35% | Block pointer 管理非连续 KV 页 |
| GQA/MQA | exp_flash_attention_gqa_001 | 40% | SIMD index select 复用 KV head |
| SDPA with Cube hints | exp_flash_attention_sdpa_001 | 35% | `dl.compile_hint` 引导 Cube 单元调度 |
| 稀疏注意力 (Prefill) | exp_flash_attention_sparse_001 | 30% | 稀疏 mask + 跳过零块计算 |

### 内存访问优化

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| Q/K/V 2D Block Pointer | exp_attn_mem_001 | 25% | `tl.make_block_ptr` 管理 Q/K/V 分块指针 |
| K/V 跨 Q 块寄存器复用 | exp_attn_mem_002 | 30% | 固定 K/V tile，滑动 Q tile |
| Q/K/V 循环 Tiling | exp_attn_mem_003 | 35% | 三层循环分块，最大化数据复用 |
| Split-K 中间结果 Workspace | exp_attn_mem_004 | 25% | Split-K 部分结果写入 workspace，最终 reduce |

### 调度优化

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 静态任务调度 | exp_attn_sched_001 | 20% | 无动态负载均衡，静态分配 head/batch 到 core |
| 变长序列处理 | exp_attn_sched_002 | 30% | `seq_offset` 索引支持 packed variable-length |
| Prefill/Decode 双路 dispatch | exp_attn_sched_003 | 25% | 根据 seq_len 动态选择 prefill 或 decode kernel |

## 关键技术

### Online Softmax 算法

标准 Softmax 需要两遍扫描（先求最大值，再归一化），无法分块。Online Softmax 通过维护运行统计量实现单遍分块计算，是 Flash Attention 内存降低到 O(S) 的核心。

**完整迭代公式**（每处理一个 K 块 `j`）：

设当前块的注意力分数向量为 `scores_j = Q_i · K_j^T / sqrt(d_k)`，则：

```
# 初始化（第 0 块之前）
m_i   = -inf        # 运行最大值
l_i   = 0.0         # 运行归一化因子（sum of exp）
acc_i = zeros(d_v)  # 运行输出累加器

# 处理第 j 块 K/V
m_ij  = max(m_i, max(scores_j))            # 更新全局最大值
p_j   = exp(scores_j - m_ij)              # 当前块的 softmax 分子（相对新最大值）
l_i   = l_i * exp(m_i - m_ij) + sum(p_j) # 缩放旧 l，加入新块贡献
acc_i = acc_i * exp(m_i - m_ij) + p_j @ V_j  # 缩放旧 acc，加入新块贡献
m_i   = m_ij                               # 更新最大值

# 最终归一化
out_i = acc_i / l_i
```

**数值稳定性**：每次更新时用 `exp(m_i - m_ij)` 缩放旧值，保证数值稳定（`m_i - m_ij ≤ 0`，exp 不溢出）。

**Triton 实现要点**：
- `m_i` 和 `l_i` 在寄存器中保持，不写回 HBM
- `acc_i` 形状为 `[BLOCK_Q, HEAD_DIM]`，保存在 L0C
- 循环结束后一次性写回：`tl.store(out_ptr, acc_i / l_i)`

### Q/K/V 分块策略

Flash Attention 的外层循环遍历 K/V 块，内层（隐式）处理 Q 块：

```
for q_block in range(0, S, BLOCK_Q):      # Q 分块（每个 program 负责一个 Q 块）
    acc, l, m = init()
    for kv_block in range(0, S, BLOCK_KV): # K/V 分块（迭代）
        K_tile = load(K, kv_block)
        V_tile = load(V, kv_block)
        scores = Q_tile @ K_tile.T * scale
        # Causal mask（可选）
        if causal:
            scores = mask_upper_triangle(scores, q_block, kv_block)
        acc, l, m = online_softmax_update(acc, l, m, scores, V_tile)
    store(out, acc / l, q_block)
```

每个 Triton program 负责一个 `(batch, head, q_block)` 三元组，grid 为 `(batch * num_heads * cdiv(S, BLOCK_Q),)`。

### GQA/MQA Head 复用

Grouped Query Attention (GQA) 中，`num_kv_heads < num_heads`，每个 KV head 对应多个 Q head。

```python
# SIMD index select 复用 KV head（exp_flash_attention_gqa_001）
kv_head_idx = q_head_idx // (num_heads // num_kv_heads)
k_ptr = K_ptr + kv_head_idx * stride_kh
v_ptr = V_ptr + kv_head_idx * stride_vh
```

避免 KV head 数据复制，直接通过索引映射实现 MQA（num_kv_heads=1）和 GQA。

### Prefill vs Decode 双路调度

- **Prefill**（S 长，通常 ≥ 512）：计算密集，使用 Flash Attention 分块路径，每核处理多个 Q 块。
- **Decode**（S = 1 或极短，auto-regressive 生成）：内存带宽瓶颈，K/V cache 较长，使用 Split-K 路径并行归约。

```python
# Dual dispatch（exp_attn_sched_003）
if seq_len > PREFILL_THRESHOLD:
    flash_attention_prefill_kernel[grid](...)
else:
    flash_attention_decode_kernel[grid](...)
```

### FP8 + UB Overflow Slicing

FP8 量化将注意力分数压缩为 8 位，减少 HBM 带宽。由于 FP8 动态范围有限，需通过 UB (Upper Bound) overflow slicing 防止 softmax 前的分数溢出：

```python
# 将大值切分为两个 FP8 范围内的片段，分别计算后合并
scores_fp8_hi = quantize_fp8(scores.clamp(max=FP8_MAX))
scores_fp8_lo = quantize_fp8((scores - scores_fp8_hi).clamp(max=FP8_MAX))
```

### 2D Block Pointer 管理 Q/K/V

```python
# Q block pointer（exp_attn_mem_001）
q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + batch_head_offset,
    shape=(S, HEAD_DIM),
    strides=(stride_sq, stride_dq),
    offsets=(q_start, 0),
    block_shape=(BLOCK_Q, HEAD_DIM),
    order=(1, 0),
)

# K/V block pointer（在 KV 循环中 advance）
k_block_ptr = tl.make_block_ptr(
    base=K_ptr + batch_kv_head_offset,
    shape=(S, HEAD_DIM),
    strides=(stride_sk, stride_dk),
    offsets=(0, 0),
    block_shape=(BLOCK_KV, HEAD_DIM),
    order=(1, 0),
)
```

### 变长序列 (`seq_offset`)

Packed variable-length 格式将不同长度的序列连续存储，用 `seq_offset` 数组记录每条序列的起始位置：

```python
# exp_attn_sched_002
seq_start = tl.load(seq_offset_ptr + batch_id)
seq_end   = tl.load(seq_offset_ptr + batch_id + 1)
actual_len = seq_end - seq_start
```

## 常见陷阱

- **忘记 Online Softmax 缩放旧 acc**：更新 `acc` 时必须乘以 `exp(m_i - m_ij)`，否则旧块贡献未被正确重新归一化，导致数值错误。
- **Causal Mask 边界错误**：在 Q block 与 KV block 的交叉处，mask 条件为 `q_pos >= kv_pos`（行 ≥ 列），容易出现 off-by-one，需以绝对序列位置计算。
- **HEAD_DIM 非 64 倍数**：Cube 单元要求 HEAD_DIM 为 64 的倍数（fp16/bf16），否则需 padding 至下一个 64 倍数。
- **GQA KV head 索引计算错误**：`num_heads` 未整除 `num_kv_heads` 时映射越界，需在 kernel 外部验证整除关系。
- **Decode 场景误用 Prefill kernel**：S=1 时 Flash Attention 分块无收益，反而因 launch overhead 变慢，应走 Split-K decode 路径。
- **FP8 未做 overflow slicing**：直接量化 softmax 前的原始分数会因 d_k 较大（scores ~ O(sqrt(d_k))）溢出 FP8 范围，导致 NaN。
- **Split-K workspace 未清零**：多次调用时 workspace 残留上一次结果，需显式清零或使用原子加法累加。
- **变长序列越界访问**：未使用 `boundary_check` 的 `tl.load` 在序列末尾会越界，必须开启边界检查或对末尾 tile 单独处理。

## 代码模板

以下为带 AutoTune 的 Flash Attention 核心 kernel（单向 causal，fp16，标准 GQA 接口）：

```python
import triton
import triton.language as tl
import torch
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': bq, 'BLOCK_KV': bkv}, num_stages=2, num_warps=4)
        for bq in [64, 128]
        for bkv in [64, 128]
    ],
    key=['S', 'HEAD_DIM'],
)
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    S, HEAD_DIM: tl.constexpr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # program 负责 (batch, q_head, q_block) 三元组
    pid      = tl.program_id(0)
    num_q_blocks = tl.cdiv(S, BLOCK_Q)
    batch_head_id = pid // num_q_blocks
    q_block_id    = pid %  num_q_blocks

    batch_id  = batch_head_id // num_heads
    q_head_id = batch_head_id %  num_heads
    # GQA: 映射到 KV head
    kv_head_id = q_head_id // (num_heads // num_kv_heads)

    q_start = q_block_id * BLOCK_Q

    # Q block pointer
    q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_id * stride_qb + q_head_id * stride_qh,
        shape=(S, HEAD_DIM), strides=(stride_qs, stride_qd),
        offsets=(q_start, 0), block_shape=(BLOCK_Q, HEAD_DIM), order=(1, 0),
    )
    # K/V block pointers
    k_base = K_ptr + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = V_ptr + batch_id * stride_vb + kv_head_id * stride_vh

    Q = tl.load(q_block_ptr, boundary_check=(0, 1))  # [BLOCK_Q, HEAD_DIM]

    # Online Softmax 状态
    m_i   = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l_i   = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc   = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # KV 上限（causal: 只看到 q_start + BLOCK_Q 之前的 KV）
    kv_end = (q_start + BLOCK_Q) if causal else S

    for kv_start in range(0, kv_end, BLOCK_KV):
        k_block_ptr = tl.make_block_ptr(
            base=k_base, shape=(S, HEAD_DIM), strides=(stride_ks, stride_kd),
            offsets=(kv_start, 0), block_shape=(BLOCK_KV, HEAD_DIM), order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_base, shape=(S, HEAD_DIM), strides=(stride_vs, stride_vd),
            offsets=(kv_start, 0), block_shape=(BLOCK_KV, HEAD_DIM), order=(1, 0),
        )
        K_tile = tl.load(k_block_ptr, boundary_check=(0, 1))  # [BLOCK_KV, HEAD_DIM]
        V_tile = tl.load(v_block_ptr, boundary_check=(0, 1))  # [BLOCK_KV, HEAD_DIM]

        # 注意力分数: [BLOCK_Q, BLOCK_KV]
        scores = tl.dot(Q, tl.trans(K_tile)) * scale

        # Causal mask
        if causal:
            q_idx  = q_start  + tl.arange(0, BLOCK_Q )[:, None]
            kv_idx = kv_start + tl.arange(0, BLOCK_KV)[None, :]
            scores = tl.where(q_idx >= kv_idx, scores, float('-inf'))

        # Online Softmax 更新
        m_ij  = tl.maximum(m_i, tl.max(scores, axis=1))          # [BLOCK_Q]
        p     = tl.exp(scores - m_ij[:, None])                    # [BLOCK_Q, BLOCK_KV]
        alpha = tl.exp(m_i - m_ij)                                # [BLOCK_Q] 缩放因子
        l_i   = l_i * alpha + tl.sum(p, axis=1)                  # [BLOCK_Q]
        acc   = acc * alpha[:, None] + tl.dot(p.to(Q.dtype), V_tile)  # [BLOCK_Q, HEAD_DIM]
        m_i   = m_ij

    # 最终归一化并写回
    out = (acc / l_i[:, None]).to(tl.float16)
    out_block_ptr = tl.make_block_ptr(
        base=Out_ptr + batch_id * stride_ob + q_head_id * stride_oh,
        shape=(S, HEAD_DIM), strides=(stride_os, stride_od),
        offsets=(q_start, 0), block_shape=(BLOCK_Q, HEAD_DIM), order=(1, 0),
    )
    tl.store(out_block_ptr, out, boundary_check=(0, 1))


def flash_attention(
    Q: torch.Tensor,   # [B, H, S, D]
    K: torch.Tensor,   # [B, Hkv, S, D]
    V: torch.Tensor,   # [B, Hkv, S, D]
    causal: bool = True,
) -> torch.Tensor:
    B, H, S, D = Q.shape
    Hkv = K.shape[1]
    assert D % 64 == 0, "HEAD_DIM 必须是 64 的倍数"
    scale = 1.0 / math.sqrt(D)
    Out = torch.empty_like(Q)
    grid = (B * H * triton.cdiv(S, 128),)  # 由 autotune 的 BLOCK_Q 决定，此处以 128 为估算
    flash_attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        S, D, scale, H, Hkv, causal,
    )
    return Out
```

## 相关页面

- [ops/matmul.md](op-class-matmul.md) — GEMM 基础（注意力中的 QK^T 和 PV 本质是 GEMM）
- [ops/softmax.md](op-class-softmax.md) — Online Softmax 算法详解
- [ops/ssm.md](op-class-ssm.md) — Linear Attention / KDA 反向优化（与 Flash Attention 共享标量消除、转置 load 合并等技术）
- techniques/tiling.md — Tiling 策略通用框架
- techniques/persistent_kernels.md — Grid-stride + 软件流水线
- techniques/operator_fusion.md — 算子融合模式（Attention+RoPE、Attention+Norm 等）
- [hardware/ascend_910b.md](ascend-910b.md) — 910b2/910b3 Cube 单元规格与 FP8 支持说明
- techniques/data_types.md — int32→fp32 比较避免 scalar fallback
- techniques/memory_access_optimization.md — 转置 load 合并化

## 反向传播经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Flash Attention 反向 KV | exp_flashattn_bwd_kv_001 | dK/dV 梯度计算，online 累积 | yaml |
| Flash Attention 反向 KV（Ascend） | exp_flashattn_bwd_kv_ascend_001 | Ascend 910b 专用路径 | yaml |
| SDPA 反向 KV | exp_sdpa_bwd_kv_001 | SDPA 反向 dK/dV | yaml |
| SDPA 反向 Q | exp_sdpa_bwd_q_001 | SDPA 反向 dQ | yaml |
| SDPA 反向归约 | exp_sdpa_bwd_reduce_001 | 反向中的 delta 归约 | yaml |
| Attention 反向 Q | exp_attn_bwd_q_001 | 基础 Attention dQ | yaml |
| Attention 反向 Q（下三角） | exp_attn_bwd_q_down_001 | 因果 mask 下 dQ | yaml |
| 稀疏注意力反向 dQ | exp_sparse_attn_backward_dq_001 | 稀疏 mask 下 dQ | yaml |
| HSTU 注意力反向 | exp_attn_bwd_hstu_001 | HSTU 架构反向 | yaml |
| Linear Attention / KDA 反向（标量消除） | exp_kda_bwd_wy_scalar_opt_001 | i64→i32、i32→fp32、block_ptr→手动指针 | yaml |
| Linear Attention / KDA 反向（结构优化） | exp_kda_bwd_wy_structure_opt_001 | Phase split、barrier 外提、LICM | yaml |

## 进阶变体经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| HSTU 注意力前向 | exp_hstu_attn_fwd_001 | HSTU（历史序列 Transformer）注意力前向 | yaml |
| HSTU 注意力调度 | exp_hstu_attn_sched_001 | HSTU 多流调度策略 | yaml |
| HSTU 注意力反向 | exp_hstu_attn_bwd_001 | HSTU 反向传播 | yaml |
| Lightning Attention | exp_lightning_attn_001 | 线性注意力 Lightning 变体 | yaml |
| Lightning Attention Decode | exp_lightning_attn_decode_001 | Lightning Attention 推理解码 | yaml |
| MLA Paged Attention | exp_mla_paged_attn_001 | Multi-Latent Attention + 分页 KV | yaml |
| Paged GQA | exp_paged_gqa_001 | GQA + paged KV cache | yaml |
| Paged KV Cache | exp_paged_kv_cache_001 | KV 缓存分页管理 | yaml |
| Paged Prefill | exp_paged_prefill_001 | 分页 prefill 阶段 | yaml |
| Decode Attention | exp_paged_attn_decode_001 | decode 阶段分页注意力 | yaml |
| QK Cube GEMM | exp_attn_qk_cube_001 | QK^T 用 Cube 加速 | yaml |
| Attention Sinks | exp_attn_sinks_001 | Attention sink 机制（StreamingLLM） | yaml |
| Attention Sinks Prefill | exp_attn_sinks_prefill_001 | Sinks prefill 阶段 | yaml |
| Attention Reduce Stage2 | exp_attn_reduce_stage2_001 | Split-K 第二阶段归约 | yaml |
| Fused Attention Preprocess | exp_fused_attn_preprocess_001 | 注意力前处理融合 | yaml |
| Causal Mask 优化 | exp_attn_causal_mask_001 | 因果 mask 跳过计算 | yaml |
| Block Pointer 设置 | exp_attn_ptr_setup_001 | QKV block pointer 配置 | yaml |
| Rope MQA | exp_rope_mqa_001 | RoPE + MQA 融合 | yaml |
| Sparse Flash Attention Prefill | exp_sparse_flash_attn_prefill_001 | 稀疏 FA prefill | yaml |
| Sparse Flash Attention Decode | exp_sparse_flash_attn_decode_001 | 稀疏 FA decode | yaml |


## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `005-_hstu_attn_fwd_one_block.py` — extracted kernel（source: `raw_experiences/extracted_kernels/005-_hstu_attn_fwd_one_block.py`）
- `006-_hstu_attn_fwd_compute.py` — extracted kernel（source: `raw_experiences/extracted_kernels/006-_hstu_attn_fwd_compute.py`）
- `007-_hstu_attn_fwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/007-_hstu_attn_fwd.py`）
- `008-_hstu_attn_bwd_one_block.py` — extracted kernel（source: `raw_experiences/extracted_kernels/008-_hstu_attn_bwd_one_block.py`）
- `009-_hstu_attn_bwd_one_col_block.py` — extracted kernel（source: `raw_experiences/extracted_kernels/009-_hstu_attn_bwd_one_col_block.py`）
- `010-_hstu_attn_bwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/010-_hstu_attn_bwd.py`）
- `011-_attn_fwd_inner.py` — extracted kernel（source: `raw_experiences/extracted_kernels/011-_attn_fwd_inner.py`）
- `012-_attn_fwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/012-_attn_fwd.py`）
- `013-grouped_attention_kernel_stage1.py` — extracted kernel（source: `raw_experiences/extracted_kernels/013-grouped_attention_kernel_stage1.py`）
- `021-backward_sum_o_do.py` — extracted kernel（source: `raw_experiences/extracted_kernels/021-backward_sum_o_do.py`）
- `025-backward_dkdv.py` — extracted kernel（source: `raw_experiences/extracted_kernels/025-backward_dkdv.py`）
- `026-backward_dq.py` — extracted kernel（source: `raw_experiences/extracted_kernels/026-backward_dq.py`）
- `055-_attn_fwd_split_cv.py` — extracted kernel（source: `raw_experiences/extracted_kernels/055-_attn_fwd_split_cv.py`）
- `069-_lightningattn_attn_decode_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/069-_lightningattn_attn_decode_kernel.py`）
- `088-_attn_fwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/088-_attn_fwd.py`）
- `173-attention_sinks_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/173-attention_sinks_kernel.py`）
- `174-attention_sinks_prefill_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/174-attention_sinks_prefill_kernel.py`）
- `175-_paged_mla_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/175-_paged_mla_fwd_kernel.py`）
- `176-_paged_gqa_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/176-_paged_gqa_fwd_kernel.py`）
- `177-_paged_gqa_fwd_kernel_high_performance.py` — extracted kernel（source: `raw_experiences/extracted_kernels/177-_paged_gqa_fwd_kernel_high_performance.py`）
- `196-sparse_flash_attention_decode_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/196-sparse_flash_attention_decode_kernel.py`）
- `199-sparse_flash_attention_prefill_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/199-sparse_flash_attention_prefill_kernel.py`）
- `216-_sdpa_infer_inner.py` — extracted kernel（source: `raw_experiences/extracted_kernels/216-_sdpa_infer_inner.py`）
- `217-_sdpa_infer_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/217-_sdpa_infer_kernel.py`）
- `218-kernel_sdpa_fwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/218-kernel_sdpa_fwd.py`）
- `219-kernel_sdpa_bwd_d.py` — extracted kernel（source: `raw_experiences/extracted_kernels/219-kernel_sdpa_bwd_d.py`）
- `220-kernel_sdpa_bwd_q.py` — extracted kernel（source: `raw_experiences/extracted_kernels/220-kernel_sdpa_bwd_q.py`）
- `221-kernel_sdpa_bwd_kv.py` — extracted kernel（source: `raw_experiences/extracted_kernels/221-kernel_sdpa_bwd_kv.py`）
- `238-_store_paged_kv_cache_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/238-_store_paged_kv_cache_kernel.py`）
- `252-causal_mask_fn.py` — extracted kernel（source: `raw_experiences/extracted_kernels/252-causal_mask_fn.py`）
- `253-_sdpa_infer_single_block.py` — extracted kernel（source: `raw_experiences/extracted_kernels/253-_sdpa_infer_single_block.py`）
- `254-paged_prefill_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/254-paged_prefill_kernel.py`）
- `255-paged_decode_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/255-paged_decode_kernel.py`）
- `256-micro_kernel_fwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/256-micro_kernel_fwd.py`）
- `257-micro_kernel_bwd_q.py` — extracted kernel（source: `raw_experiences/extracted_kernels/257-micro_kernel_bwd_q.py`）
- `258-micro_kernel_bwd_kv.py` — extracted kernel（source: `raw_experiences/extracted_kernels/258-micro_kernel_bwd_kv.py`）
- `259-kernel_sda_fwd_up.py` — extracted kernel（source: `raw_experiences/extracted_kernels/259-kernel_sda_fwd_up.py`）
- `260-kernel_sda_fwd_down.py` — extracted kernel（source: `raw_experiences/extracted_kernels/260-kernel_sda_fwd_down.py`）
- `261-kernel_sda_bwd_d.py` — extracted kernel（source: `raw_experiences/extracted_kernels/261-kernel_sda_bwd_d.py`）
- `262-kernel_sda_bwd_q_up.py` — extracted kernel（source: `raw_experiences/extracted_kernels/262-kernel_sda_bwd_q_up.py`）
- `263-kernel_sda_bwd_q_down.py` — extracted kernel（source: `raw_experiences/extracted_kernels/263-kernel_sda_bwd_q_down.py`）
- `264-kernel_sda_bwd_kv_left.py` — extracted kernel（source: `raw_experiences/extracted_kernels/264-kernel_sda_bwd_kv_left.py`）
- `265-kernel_sda_bwd_kv_right.py` — extracted kernel（source: `raw_experiences/extracted_kernels/265-kernel_sda_bwd_kv_right.py`）
- `053-get_params.py` — extracted kernel（source: `raw_experiences/extracted_kernels/053-get_params.py`）

<!-- okf:related:start -->

# 相关

- 实现示例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention, gqa
- 实现示例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../ops/block-sparse-attention.md) — 主题相关：共享 attention, flash_attention, gqa
- 实现示例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../ops/flash-attention-score-grad.md) — 主题相关：共享 attention, flash_attention, softmax
- 实现示例: [FlashAttentionScore training-forward on Ascend NPU](../ops/flash-attention-score.md) — 主题相关：共享 attention, flash_attention, softmax
- 实现示例: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../ops/fused-floyd-attention-grad.md) — 主题相关：共享 attention
- 实现示例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../ops/fused-floyd-attention.md) — 主题相关：共享 attention, softmax
- 实现示例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../ops/fused-infer-attention-score.md) — 主题相关：共享 attention, flash_attention
- 实现示例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../ops/lightning-indexer.md) — 主题相关：共享 attention, gqa
- 实现示例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention, gqa
- 实现示例: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../ops/nsa-selected-attention.md) — 主题相关：共享 attention, gqa
- 实现示例: [PromptFlashAttention inference/prefill on Ascend NPU](../ops/prompt-flash-attention.md) — 主题相关：共享 attention, flash_attention, softmax
- 实现示例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../ops/rmsnorm-rope.md) — 同属优化知识，主题相关
- 实现示例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../ops/sparse-flash-attention.md) — 主题相关：共享 attention, flash_attention, softmax
- 讲解接口: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 讲解接口: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention
- 讲解接口: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 同属优化知识，主题相关
- 讲解接口: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention, flash_attention
- 配套样例: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, attention, gqa
- 配套样例: [FlashAttention 变体的内存访问优化](../runbooks/algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, attention, flash_attention, gqa
- 配套样例: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../runbooks/algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3, attention, flash_attention
- 配套样例: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, softmax
- 配套样例: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../runbooks/compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, flash_attention, softmax
- 配套样例: [Softmax Backward 内存访问优化聚合](../runbooks/memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, softmax
- 配套样例: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../runbooks/meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, attention, softmax
- 配套样例: [Flash Attention Causal Skeleton](skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 配套样例: [Rmsnorm Fused Skeleton](skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, attention, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, operator_class, softmax
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, operator_class, softmax
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
