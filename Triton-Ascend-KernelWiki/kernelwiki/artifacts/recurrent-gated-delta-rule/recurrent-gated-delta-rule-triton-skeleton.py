# recurrent_gated_delta_rule - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现。reproducibility: pseudocode。
# 表达设计意图：线性注意力 RGDR（循环门控 Delta 规则），SSM 风格状态更新。
# AIV-only，无 Cube 参与。来源：attention/recurrent_gated_delta_rule/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def recurrent_gated_delta_rule_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, state_ptr, out_ptr, state_out_ptr,
    cu_seqlens_ptr, ssm_state_indices_ptr,
    g_ptr, gk_ptr, num_accepted_tokens_ptr,
    # shape
    batch, total_tokens,
    # meta (constexpr)
    NK: tl.constexpr, NV: tl.constexpr,  # head dims
    DK: tl.constexpr, DV: tl.constexpr,  # feature dims
    HAS_G: tl.constexpr, HAS_GK: tl.constexpr, HAS_ACCEPTED: tl.constexpr,
    BLOCK_S: tl.constexpr,  # S 状态分块
    SCALE: tl.constexpr,     # scale_value
):
    pid = tl.program_id(0)

    # 门控系数
    g = tl.load(g_ptr) if HAS_G else 0.0
    gk = tl.load(gk_ptr) if HAS_GK else 0.0
    alpha = tl.exp(g).to(tl.float32)
    alpha_k = tl.exp(gk).to(tl.float32)

    # 分核：每核处理一个 batch 的串行 token 序列
    batch_idx = pid // (NK // BLOCK_S)
    head_start = (pid % (NK // BLOCK_S)) * BLOCK_S
    offs_head = head_start + tl.arange(0, BLOCK_S)
    h_mask = offs_head < NK

    # 初始化 S 状态 [NV, DK]（每核独立份）
    S = tl.zeros([DV, DK], dtype=tl.float32)

    # 串行遍历 token（cu_seqlens 前缀和得 batch 范围）
    seq_start = tl.load(cu_seqlens_ptr + batch_idx)
    seq_end = tl.load(cu_seqlens_ptr + batch_idx + 1)
    for t in range(seq_start, seq_end):
        # 当前步的 q/k/v/beta [DK], [DV]
        q = tl.load(q_ptr + t * NK * DK + offs_head[:, None] * DK + tl.arange(0, DK)[None, :],
                    mask=h_mask[:, None], other=0.0).to(tl.float32)
        # 简化：顺序读取，实际按 stride 寻址

        # 门控衰减
        decayed = alpha * alpha_k * S  # [DV, DK]

        # 误差: v - decayed·k
        k = tl.load(k_ptr + ...).to(tl.float32)  # [DK]
        v = tl.load(v_ptr + ...).to(tl.float32)  # [DV]
        attn = tl.dot(decayed, k)  # [DV]
        error = v - attn

        # 更新 S
        beta = tl.load(beta_ptr + ...).to(tl.float32)  # scalar
        S = decayed + beta * tl.dot(error[:, None], k[None, :])  # [DV, DK] 外积

        # 输出
        o = tl.dot(S, q) * SCALE  # [DV]
        tl.store(out_ptr + t * DV + tl.arange(0, DV), o)

    # 写回最终状态
    tl.store(state_out_ptr + batch_idx * NK * DV * ..., S)


# grid = (batch * NK,)
# 注意：这是高度简化的伪代码。实际 CANN 中 S_t 更新公式有 Diag(αk) 逐元素门控和
#   β*(v - α·Diag(αk)·S_{t-1}·k)·k^T 外积项，Triton 中需用 `tl.dot` 处理。
# 变长序列由 cu_seqlens 前缀和驱动，CANN 的 ssm_state_indices 管理多 batch 状态切片。
# AIV-only 意味着无 grid 分核，全在 Vector 上串行。