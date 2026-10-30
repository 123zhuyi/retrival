# fused_floyd_attention_grad - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：Floyd Attention 反向梯度（双流 K1/V1+K2/V2）
#   6 个 matmul 反向：dQ/dK1/dV1/dK2/dV2
# CANN 专有机制（三段式 SyncAll / ND↔NZ 格式转换）不在此体现。
# 来源：attention/fused_floyd_attention_grad/ (commit d28d8f9...), README.md.

import triton
import triton.language as tl


@triton.jit
def fused_floyd_attention_grad_kernel(
    # pointers
    q_ptr, k1_ptr, v1_ptr, k2_ptr, v2_ptr,
    dy_ptr,                     # 下游梯度 dY
    p_ptr,                      # 前向 softmax 输出 P = softmax(QK^T/sqrt(d))
    dq_ptr, dk1_ptr, dv1_ptr, dk2_ptr, dv2_ptr,
    # strides
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_k1b, stride_k1n, stride_k1s, stride_k1d,
    stride_v1b, stride_v1n, stride_v1s, stride_v1d,
    stride_k2b, stride_k2n, stride_k2s, stride_k2d,
    stride_v2b, stride_v2n, stride_v2s, stride_v2d,
    # shape / scalars
    sm_scale,
    q_len, kv_len,
    # block meta (constexpr)
    BLOCK_M: tl.constexpr,  # S1 基本块
    BLOCK_N: tl.constexpr,  # S2 基本块
    D: tl.constexpr,        # head dim (32/64/128)
):
    pid = tl.program_id(0)  # 沿 BN2GS1 轴切分

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    m_mask = offs_m < q_len
    n_mask = offs_n < kv_len

    # Load Q, K1, V1, K2, V2
    q = tl.load(q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None], other=0.0).to(tl.float32)

    k1 = tl.load(k1_ptr + offs_n[:, None] * stride_k1s + offs_d[None, :] * stride_k1d,
                 mask=n_mask[:, None], other=0.0)
    v1 = tl.load(v1_ptr + offs_n[:, None] * stride_v1s + offs_d[None, :] * stride_v1d,
                 mask=n_mask[:, None], other=0.0)
    k2 = tl.load(k2_ptr + offs_n[:, None] * stride_k2s + offs_d[None, :] * stride_k2d,
                 mask=n_mask[:, None], other=0.0)
    v2 = tl.load(v2_ptr + offs_n[:, None] * stride_v2s + offs_d[None, :] * stride_v2d,
                 mask=n_mask[:, None], other=0.0)

    # 前向 P = softmax(scale * (Q·K1^T + Q·K2^T), mask)
    # 简化：假设 P 已保存（作为 p_ptr 输入）
    p = tl.load(p_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    # dY 加载
    dy = tl.load(dy_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                 mask=m_mask[:, None], other=0.0).to(tl.float32)

    # ---- Pre: dS = P * dY ----
    dS = p * tl.sum(dy[:, None, :] * q[:, :, None], axis=2)  # 伪过程：简化 P 和 dY 的 dot 关系
    # 实际：dS = P * (dY · Q^T) 等更复杂的链式法则

    # ---- dV1 = P^T · dY ----
    dv1 = tl.dot(p.trans(1, 0), dy)  # [BLOCK_N, D]

    # ---- dV2 = P^T · dY ----
    dv2 = tl.dot(p.trans(1, 0), dy)  # [BLOCK_N, D]

    # ---- dQ = (dS·K1 + dS·K2) / sqrt(d) ----
    dq = (tl.dot(dS, k1) + tl.dot(dS, k2)) / tl.sqrt(D)

    # ---- dK1 = dS^T · Q / sqrt(d) ----
    dk1 = tl.dot(dS.trans(1, 0), q) / tl.sqrt(D)

    # ---- dK2 = dS^T · Q / sqrt(d) ----
    dk2 = tl.dot(dS.trans(1, 0), q) / tl.sqrt(D)

    # store gradients
    tl.store(dq_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
             dq.to(dq_ptr.dtype.element_ty()), mask=m_mask[:, None])
    tl.store(dk1_ptr + offs_n[:, None] * stride_k1s + offs_d[None, :] * stride_k1d,
             dk1.to(dk1_ptr.dtype.element_ty()), mask=n_mask[:, None])
    tl.store(dv1_ptr + offs_n[:, None] * stride_v1s + offs_d[None, :] * stride_v1d,
             dv1.to(dv1_ptr.dtype.element_ty()), mask=n_mask[:, None])
    tl.store(dk2_ptr + offs_n[:, None] * stride_k2s + offs_d[None, :] * stride_k2d,
             dk2.to(dk2_ptr.dtype.element_ty()), mask=n_mask[:, None])
    tl.store(dv2_ptr + offs_n[:, None] * stride_v2s + offs_d[None, :] * stride_v2d,
             dv2.to(dv2_ptr.dtype.element_ty()), mask=n_mask[:, None])

# grid = (B * N2 * G * ceil(S1 / BLOCK_M),)  # 沿 BN2GS1 轴切分
# 注意：dS 的精确公式依赖前向 softmax 保存的 max/sum 恢复，此处简化