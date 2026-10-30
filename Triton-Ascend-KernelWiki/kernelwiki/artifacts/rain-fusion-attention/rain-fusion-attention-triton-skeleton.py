# rain_fusion_attention - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现。reproducibility: pseudocode。
# 表达设计意图：CATLASS 风格块级稀疏 FlashAttention（selectIdx 驱动）。
# 来源：attention/rain_fusion_attention/ (d28d8f9...), README L138-163.

import triton
import triton.language as tl


@triton.jit
def rain_fusion_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    select_idx_ptr, select_num_idx_ptr,
    atten_mask_ptr,
    block_shape_x, block_shape_y,
    sm_scale,
    num_heads, head_dim,
    # meta (constexpr)
    BLOCK_M: tl.constexpr,  # blockShapeX
    BLOCK_N: tl.constexpr,  # blockShapeY
    D: tl.constexpr,
    SM_PRECISION: tl.constexpr,  # 0=float32, 1=half
    OUTPUT_LSE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 沿 Q 块 + head 分核
    q_block_idx = pid // num_heads
    head_idx = pid % num_heads

    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :]).to(tl.float32)  # [BLOCK_M, D]

    # 在线 softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # 稀疏 KV 块循环（selectIdx 驱动）
    num_kv_blocks = tl.load(select_num_idx_ptr + q_block_idx * num_heads + head_idx)
    for b in range(num_kv_blocks):
        kv_block_idx = tl.load(select_idx_ptr + q_block_idx * num_heads * num_kv_blocks
                               + head_idx * num_kv_blocks + b)
        offs_n = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        k = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :]).to(tl.float32)
        v = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :]).to(tl.float32)

        # QK
        s = tl.dot(q, k.trans(1, 0)).to(tl.float32) * sm_scale

        # mask（可选）
        if atten_mask_ptr:
            mask = tl.load(atten_mask_ptr + offs_m[:, None] * num_kv_blocks + offs_n[None, :])
            s = tl.where(mask != 0, -float("inf"), s)

        # online softmax
        m_ij = tl.max(s, axis=1)
        s = s - m_ij[:, None]
        p = tl.exp(s).to(v.dtype)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, v).to(tl.float32)
        m_i = tl.maximum(m_i, m_ij)

    # rescale + 写回
    acc = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], acc)

    if OUTPUT_LSE:
        lse = tl.log(l_i) + m_i
        tl.store(lse_ptr + offs_m * num_heads + head_idx, lse)


# grid = (num_q_blocks * num_heads,)
# SM_PRECISION=1(half) 时，softmax 计算用 half 而非 float32（精度换性能）。
# CATLASS 模板库机制（CANN 专有）不在此体现。
# 注意：README 说 BNSD 不支持，但 tilingkey.h 和 kernel 入口有 BNSD 分支，以源码为准。