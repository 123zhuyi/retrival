#!/usr/bin/env python3
# """
# FusedFloydAttention - Triton-Ascend skeleton
#
# 知识制品 (knowledge artifact only)：表达设计意图，非可编译移植。
# reproducibility: pseudocode。
#
# 公式（README.md L19-25）：
#   weights = Softmax(attenMask + scale * (Q@K1^T + Q@K2^T))
#   attention_out = weights@V1 + weights@V2
# """
import triton
import triton.language as tl


@triton.jit
def _fused_floyd_attn_kernel(
    q_ptr, k1_ptr, v1_ptr, k2_ptr, v2_ptr, atten_mask_ptr,
    attn_out_ptr, softmax_max_ptr, softmax_sum_ptr,
    sm_scale, S1, S2, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < S1
    mask_n = offs_n < S2

    q = tl.load(q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=mask_m[:, None], other=0.0).to(tl.float32)
    k1 = tl.load(k1_ptr + offs_n[:, None] * stride_k1s + offs_d[None, :] * stride_k1d,
                 mask=mask_n[:, None], other=0.0).to(tl.float32)
    k2 = tl.load(k2_ptr + offs_n[:, None] * stride_k2s + offs_d[None, :] * stride_k2d,
                 mask=mask_n[:, None], other=0.0).to(tl.float32)

    # QK1^T + QK2^T
    s = (tl.dot(q, tl.trans(k1)) + tl.dot(q, tl.trans(k2))) * sm_scale

    if HAS_MASK:
        mask = tl.load(atten_mask_ptr + ...)
        s = tl.where(mask, -float("inf"), s)

    # online softmax
    m_i = tl.max(s, axis=1)
    p = tl.exp(s - m_i[:, None])
    l_i = tl.sum(p, axis=1)
    tl.store(softmax_max_ptr + ... , m_i, mask=mask_m)
    tl.store(softmax_sum_ptr + ... , l_i, mask=mask_m)

    # P@V1 + P@V2
    v1 = tl.load(v1_ptr + offs_n[:, None] * stride_v1s + offs_d[None, :] * stride_v1d,
                 mask=mask_n[:, None], other=0.0)
    v2 = tl.load(v2_ptr + offs_n[:, None] * stride_v2s + offs_d[None, :] * stride_v2d,
                 mask=mask_n[:, None], other=0.0)
    out = tl.dot(p_norm, v1) + tl.dot(p_norm, v2)
    tl.store(attn_out_ptr + ... , out.to(attn_out_ptr.dtype.element_ty), mask=mask_m[:, None])