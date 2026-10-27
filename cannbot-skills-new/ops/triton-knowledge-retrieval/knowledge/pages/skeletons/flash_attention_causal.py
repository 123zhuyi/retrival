"""
骨架：Flash Attention（因果掩码，支持 GQA/MQA）
适用：seq_len≥1024, fp16/bf16
经验：exp_fused_attention_002 (50%), exp_flash_attention_gqa_001 (40%)
参考：ops/attention.md, techniques/online_algorithms.md
约束：[hardware/constraints_cheatsheet]
"""

import torch
import triton
import triton.language as tl


configs = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_stages=2, num_warps=4)
    for bm in [64, 128]
    for bn in [64, 128]
]


@triton.autotune(configs=configs, key=['SEQ_LEN', 'HEAD_DIM'])
@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,   # GQA: KV head 数（< Q head 数时为 MQA/GQA）
    NUM_Q_HEADS: tl.constexpr,
    SCALE: tl.constexpr,           # 1.0 / sqrt(HEAD_DIM)
    CAUSAL: tl.constexpr,          # 因果掩码开关
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # pid_m: 负责哪个 Q block；pid_bh: batch * num_q_heads
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // NUM_Q_HEADS
    q_head_id = pid_bh % NUM_Q_HEADS
    # GQA：多个 Q head 共享同一个 KV head
    kv_head_id = q_head_id * NUM_KV_HEADS // NUM_Q_HEADS

    # ── Q tile 偏移 ──────────────────────────────────────────────
    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + batch_id * stride_qb + q_head_id * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)  # [BLOCK_M, HEAD_DIM]

    # ── Online Softmax 初始化 ────────────────────────────────────
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running sum(exp)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)        # output accumulator

    k_base = k_ptr + batch_id * stride_kb + kv_head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + kv_head_id * stride_vh

    # 因果掩码：Q block 只看 ≤ q_start+BLOCK_M 的 KV
    kv_end = (q_start + BLOCK_M) if CAUSAL else SEQ_LEN

    # ── 对每个 KV block 做 online softmax ────────────────────────
    for j in range(0, tl.cdiv(kv_end, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K tile: [HEAD_DIM, BLOCK_N]
        k_ptrs = k_base + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < SEQ_LEN, other=0.0)

        # QK^T scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k) * SCALE

        # 因果掩码：score[i,j] where i < j → -inf
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            scores = tl.where(causal_mask, scores, float('-inf'))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))          # [BLOCK_M]
        p = tl.exp(scores - m_ij[:, None])                       # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, axis=1)                                 # [BLOCK_M]

        alpha = tl.exp(m_i - m_ij)                               # correction factor
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V tile: [BLOCK_N, HEAD_DIM]
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # ── 最终归一化 ────────────────────────────────────────────────
    out = (acc / l_i[:, None]).to(tl.float16)

    # ── Store ─────────────────────────────────────────────────────
    o_base = out_ptr + batch_id * stride_ob + q_head_id * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out, mask=offs_m[:, None] < SEQ_LEN)


def flash_attention(
    q: torch.Tensor,   # [B, H_q, S, D]
    k: torch.Tensor,   # [B, H_kv, S, D]
    v: torch.Tensor,   # [B, H_kv, S, D]
    causal: bool = True,
) -> torch.Tensor:
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    scale = D ** -0.5
    out = torch.empty_like(q)

    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_M']), B * H_q)
    flash_attention_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        SEQ_LEN=S,
        HEAD_DIM=D,
        NUM_KV_HEADS=H_kv,
        NUM_Q_HEADS=H_q,
        SCALE=scale,
        CAUSAL=causal,
    )
    return out
