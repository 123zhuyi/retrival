# fused_infer_attention_score - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：统一 prefill（Q_S>1，BLOCK_M>1）+ decode（Q_S=1，BLOCK_M=1，FlashDecode）FlashAttention 前向，
#   在线 softmax，可选 mask/pse/PA/sharedPrefix/RoPE/quant/softmaxLse 分支。
# CANN 专有机制（attn_infra 的 layout ND↔NZ / arch buffer·sync / gemm Cube L1·L0 物理搬运 / MTE / queue·event）
#   不在此体现，抽象为 tile 级语义。
# 来源：attention/fused_infer_attention_score/ (commit d28d8f9...), docs/aclnnFusedInferAttentionScoreV5.md.
#
# attn_infra 分层映射（见 artifact pipeline-diagram.md §1 / tiling_dispatch_tables.md §H）：
#   gemm/block_mmad_qk  -> tl.dot(q, k.T)         # Bmm1
#   gemm/block_mmad_pv  -> tl.dot(p, v)           # Bmm2
#   epilogue/block_epilogue_online_softmax -> running max/sum + exp(m_old-m_new) rescale
#   epilogue/block_epilogue_rescale_o      -> o = acc / l_i
#   epilogue/tile_common (elemwise)        -> pse/scale/mask
#   layout/matrix (ND/NZ)                  -> Triton layout（编译器管理，不迁移）
#   arch/resource, cross_core_sync         -> Triton 编译器管理（不迁移）
#
#   program_id(0) <- CANN core id 沿 BN2 轴
#   program_id(1) <- FlashDecode split-Kv 维（仅 decode + BN2 不足时）
#   BLOCK_M <- Q_S（prefill 多 token / decode=1）
#   BLOCK_N <- S2 基本块（值 source-reported，needs_verification）

import triton
import triton.language as tl


# ========== split kernel：统一 prefill / decode ==========
# prefill: BLOCK_M = Q_S（多 token），单 kernel 沿 S2 全长在线 FlashSoftmax
# decode : BLOCK_M = 1，FlashDecode split-Kv（program_id(1) 切 S2），combine 阶段规约
@triton.jit
def fia_split_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    pse_ptr, atten_mask_ptr,
    block_table_ptr,            # PagedAttention
    key_prefix_ptr, value_prefix_ptr,   # sharedPrefix
    # strides (省略，按 layout 计算)
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_vb, stride_vn, stride_vs, stride_vd,
    stride_ob, stride_on, stride_os, stride_od,
    # shape / scalars
    sm_scale,
    q_len, kv_len,              # actualSeqLengths / actualSeqLengthsKv
    q_start_idx, kv_start_idx,  # V5 qStartIdx/kvStartIdx
    kv_padding,                 # query/kv padding
    # feature flags (constexpr)
    HAS_PSE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_PA: tl.constexpr,       # PagedAttention
    USE_PREFIX: tl.constexpr,   # sharedPrefix
    USE_ROPE: tl.constexpr,     # RoPE 解耦（Q/K 拼接 nope+rope）
    WRITE_LSE: tl.constexpr,    # softmaxLse 输出
    DECODE: tl.constexpr,       # decode (Q_S=1) vs prefill (Q_S>1)
    # block meta (constexpr) -- 值 source-reported，非最优声明
    BLOCK_M: tl.constexpr,      # Q_S：prefill 多 token / decode=1
    BLOCK_N: tl.constexpr,      # S2 基本块
    D_NOPE: tl.constexpr,       # nope 维（RoPE 解耦时）
    D_ROPE: tl.constexpr,       # rope 维（RoPE 解耦时；无 RoPE 时 D_ROPE=0）
    DV: tl.constexpr,
    KV_SPLIT: tl.constexpr,     # FlashDecode 每 split 的 BLOCK_N 数（仅 decode）
):
    pid_bn = tl.program_id(0)       # BN2 维
    pid_split = tl.program_id(1)    # FlashDecode split 维（prefill 恒 0）

    offs_m = pid_bn * BLOCK_M + tl.arange(0, BLOCK_M)  # 简化：BN2 含 B/N2，M 维=Q_S
    offs_d = tl.arange(0, D_NOPE + D_ROPE)
    m_mask = offs_m < q_len

    # Q 块 [BLOCK_M, D]，FP32 累加（RoPE 解耦时 D = nope + rope 拼接）
    q = tl.load(q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None], other=0.0).to(tl.float32)

    # 在线 softmax 状态（attn_infra epilogue online_softmax）
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)

    # S2 区间：prefill 全长 / decode 本 split 子段
    if DECODE:
        split_start = pid_split * KV_SPLIT * BLOCK_N
        split_end = tl.minimum(split_start + KV_SPLIT * BLOCK_N, kv_len)
    else:
        split_start = 0
        split_end = kv_len

    # ---- sharedPrefix：前缀 KV 先参与（可选）----
    if USE_PREFIX:
        acc, m_i, l_i = _attend_kv_range(q, key_prefix_ptr, value_prefix_ptr,
                                         0, prefix_len, ...)  # 伪过程

    # S2 分块循环（attn_infra gemm Bmm1 + epilogue softmax + gemm Bmm2）
    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < kv_len

        # PagedAttention：block_table 查物理 KV 偏移
        if USE_PA:
            kv_offs = _pa_resolve(block_table_ptr, offs_n)   # 伪过程，needs_verification
        else:
            kv_offs = offs_n + kv_start_idx

        k = tl.load(k_ptr + kv_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_offs[:, None] * stride_vs + tl.arange(0, DV)[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)

        # Bmm1: QK^T（attn_infra gemm block_mmad_qk），FP32 累加
        s = tl.dot(q, k.trans(1, 0)).to(tl.float32) * sm_scale   # [BLOCK_M, BLOCK_N]

        # pse（位置编码，pseType 决定 add/mul 顺序）
        if HAS_PSE:
            pse = tl.load(pse_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
                          mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            s = s + pse   # pseType 示意

        # atten_mask select -> 负极小
        if HAS_MASK:
            mask = tl.load(atten_mask_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
                           mask=m_mask[:, None] & n_mask[None, :], other=0)
            s = tl.where(mask != 0, -float("inf"), s)

        # 在线 FlashSoftmax（attn_infra epilogue block_epilogue_online_softmax）
        m_ij = tl.max(s, axis=1)
        s = s - m_ij[:, None]
        p = tl.exp(s).to(v.dtype)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)                      # rescale factor
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, v).to(tl.float32)   # Bmm2 + PV rescale
        m_i = tl.maximum(m_i, m_ij)

    # rescale 写回（attn_infra epilogue block_epilogue_rescale_o）
    acc = acc / l_i[:, None]
    o = acc.to(o_ptr.dtype.element_ty())
    tl.store(o_ptr + offs_m[:, None] * stride_os + tl.arange(0, DV)[None, :] * stride_od,
             o, mask=m_mask[:, None])

    # softmaxLse 输出（softmaxLseFlag；decode 供 combine，prefill 供下游）
    if WRITE_LSE:
        lse = tl.log(l_i) + m_i    # needs_verification：精确公式以 block_epilogue_online_softmax.hpp 为准
        tl.store(lse_ptr + offs_m, lse, mask=m_mask)


# ========== FlashDecode combine kernel（仅 decode） ==========
# 各 split 的 partial lse / partial o 加权规约（同 IFA FlashDecode）
@triton.jit
def fia_combine_kernel(
    partial_o_ptr, partial_lse_ptr, attention_out_ptr,
    num_splits: tl.constexpr, DV: tl.constexpr,
):
    pid_bn = tl.program_id(0)
    lse = tl.load(partial_lse_ptr + pid_bn * num_splits + tl.arange(0, num_splits))
    scale = tl.exp(lse - tl.max(lse, axis=0))
    scale = scale / tl.sum(scale, axis=0)
    res = tl.zeros([DV], dtype=tl.float32)
    for j in range(num_splits):
        split_o = tl.load(partial_o_ptr + (pid_bn * num_splits + j) * DV + tl.arange(0, DV))
        res += split_o.to(tl.float32) * scale[j]
    tl.store(attention_out_ptr + pid_bn * DV + tl.arange(0, DV),
             res.to(attention_out_ptr.dtype.element_ty))


# grid_split   = (B * N2, num_splits if DECODE else 1)
# grid_combine = (B * N2,)            # 仅 decode

# ---- 注记分支（不在此展开实现）----
# RoPE 解耦（USE_ROPE）：Q/K 沿 D 拼接 nope+rope，s = tl.dot([q_nope,q_rope], [k_nope,k_rope].T)；
#   V 仅用 nope 部分（DV = D_NOPE）。
# 分离 KV 反量化（V5）：K/V 各自 cast + keyAntiquantScale/valueAntiquantScale 融合（per-token/head）。
# learnableSink：sink 项加入 attention 分母（needs_verification 精确公式）。
# 空 query/KV：q_len==0 -> early return；kv_len==0 -> 输出全零。
