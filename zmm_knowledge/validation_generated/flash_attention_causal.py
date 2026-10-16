"""
Generated validation kernel: Causal FlashAttention forward on Ascend 910B
Retrieved knowledge:
- runbooks/algorithm-numerics/traj-attention-memory_access.md -> Online Softmax + block pointer + KV cache
- runbooks/algorithm-numerics/online-softmax-flash-attention.md -> online softmax update formula
- runbooks/meta-skills/traj-attention-workflow_opt.md -> standard attention workflow / causal mask
"""
import math
import triton
import triton.language as tl
import torch


@triton.jit
def flash_attention_causal_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    B,
    H,
    S,
    D: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # grid: (num_q_blocks, B*H)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    stride_qb = H * S * D
    stride_qh = S * D
    stride_qs = D

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_qb + head_idx * stride_qh
    v_base = v_ptr + batch_idx * stride_qb + head_idx * stride_qh
    o_base = out_ptr + batch_idx * stride_qb + head_idx * stride_qh

    # Q tile offsets
    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    q = tl.load(
        q_base + q_offs[:, None] * stride_qs + d_offs[None, :],
        mask=(q_offs[:, None] < S) & (d_offs[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    # running max / sum / acc
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    kv_offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        # K tile
        k = tl.load(
            k_base + (start_n + kv_offs_n)[:, None] * stride_qs + d_offs[None, :],
            mask=((start_n + kv_offs_n)[:, None] < S) & (d_offs[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_M, BLOCK_N)

        # causal mask: q_pos >= kv_pos
        q_pos = q_offs[:, None]
        kv_pos = (start_n + kv_offs_n)[None, :]
        scores = tl.where(q_pos >= kv_pos, scores, float("-inf"))

        # online softmax
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # V tile
        v = tl.load(
            v_base + (start_n + kv_offs_n)[:, None] * stride_qs + d_offs[None, :],
            mask=((start_n + kv_offs_n)[:, None] < S) & (d_offs[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        o_base + q_offs[:, None] * stride_qs + d_offs[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=(q_offs[:, None] < S) & (d_offs[None, :] < D),
    )


def flash_attention_causal(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Q/K/V: [B, H, S, D]"""
    B, H, S, D = Q.shape
    out = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(D)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    flash_attention_causal_kernel[grid](
        Q, K, V, out,
        B, H, S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


if __name__ == "__main__":
    B, H, S, D = 2, 8, 1024, 64
    q = torch.randn(B, H, S, D, dtype=torch.float16, device="cpu")
    k = torch.randn(B, H, S, D, dtype=torch.float16, device="cpu")
    v = torch.randn(B, H, S, D, dtype=torch.float16, device="cpu")
    o = flash_attention_causal(q, k, v)
    print("FlashAttention output shape:", o.shape)
