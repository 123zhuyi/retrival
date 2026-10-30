#!/usr/bin/env python3
# """
# FlashAttentionScoreGrad (反向) - Triton-Ascend skeleton
#
# 知识制品 (knowledge artifact only)：表达设计意图，非可编译移植。
# 未编译、未在 NPU 运行、未 benchmark。reproducibility: pseudocode。
#
# 映射 CANN FlashAttentionScoreGrad 6 步反向（docs/FAG算子设计介绍.md L9-14）：
#   1. 重算 p = SimpleSoftmax(Mask(QK^T + pse) * scale)
#   2. dp = Dropout(dY @ V^T)
#   3. ds = p * (dp - FlashSoftmaxGrad(dY, Y))   # = p_norm * (dP - rowsum(dP * p_norm))
#   4. dq = (ds @ K) * scale
#   5. dk = (ds^T @ Q) * scale
#   6. dv = Dropout(p)^T @ dY
#
# CANN -> Triton-Ascend 映射：
#   - 核间切分轴 (B,N2,G,S1) / block_dim  -> grid + tl.program_id
#   - S1.i / S2.i / D  CV 基本块          -> tl.constexpr BLOCK_M / BLOCK_N / D
#   - 尾块 (blockNum*BLOCK < total)        -> mask = offs < n 的 masked load/store
#   - FP32 累加 (L0C/UB FP32 中间)         -> tl.dot(...).to(tl.float32) 累加器
#   - 重算 p 的 mm1 workspace              -> on-chip tl tensor / drop (Triton 管理 temp)
#   - MTE/BufferID/queue/event/fixp_l0c_to_ub -> 抽象为 tile 内复用 + fused load/store，不移植
#   - 空 tensor 100MB 清零路径              -> 启动器短路（此处不表达）
#
# 基本块数值（BLOCK_M=128, BLOCK_N=128/1024）为源码报告值，非最优主张。
# dK/dV 跨 query 块累加用 atomic；dQ 在 program 内累加后写回。
# """
import triton
import triton.language as tl


@triton.jit
def _flash_attn_bwd_dq_dk_dv(
    q_ptr, k_ptr, v_ptr, dy_ptr, attn_in_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    sm_scale,
    # strides (按 BNSD 由 launcher 计算；TND/BSH 等经布局解析后同样以 stride 表达)
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_vb, stride_vn, stride_vs, stride_vd,
    stride_dyb, stride_dyn, stride_dys, stride_dyd,
    stride_dqb, stride_dqn, stride_dqs, stride_dqd,
    stride_dkb, stride_dkn, stride_dks, stride_dkd,
    stride_dvb, stride_dvn, stride_dvs, stride_dvd,
    Sq, Skv,
    BLOCK_M: tl.constexpr,   # = S1.i  (arch35: 128; arch32 V 侧: 8/64)
    BLOCK_N: tl.constexpr,   # = S2.i  (arch35: 128; arch32 V 侧: 1024)
    D: tl.constexpr,         # head dim
    DV: tl.constexpr,        # value head dim (可 != D)
    HAS_MASK: tl.constexpr,
    HAS_PSE: tl.constexpr,
    HAS_DROP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # 一个 program 处理一个 (b, n2* g, s1_block)；与 CANN 核间切 (B,N2,G,S1) 对应。
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)   # n2 * g 折叠
    pid_m = tl.program_id(2)   # S1 块索引

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_dv = tl.arange(0, DV)
    mask_m = offs_m < Sq

    # Q [BLOCK_M, D], dY [BLOCK_M, DV]  (步 2/4/6 的左矩阵)
    q_ptrs = q_ptr + pid_b * stride_qb + pid_n * stride_qn + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    dy_ptrs = dy_ptr + pid_b * stride_dyb + pid_n * stride_dyn + offs_m[:, None] * stride_dys + offs_dv[None, :] * stride_dyd
    dy = tl.load(dy_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # dQ 在 program 内累加 (步 4)；dK/dV 跨 query 块用 atomic (步 5/6)
    acc_dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # 在线 softmax 重算 P (步 1)：running max/sum + rescale
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # 注：完整实现需两遍——先算完整 softmax 统计再算 ds。此处为表达意图的合并伪代码。

    for start_n in range(0, Skv, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Skv

        # K [BLOCK_N, D], V [BLOCK_N, DV]
        k = tl.load(k_ptr + pid_b * stride_kb + pid_n * stride_kn + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                     mask=mask_n[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + pid_b * stride_vb + pid_n * stride_vn + offs_n[:, None] * stride_vs + offs_dv[None, :] * stride_vd,
                     mask=mask_n[:, None], other=0.0).to(tl.float32)

        # 步 1: s = QK^T * scale (FP32 累加)
        s = tl.dot(q, tl.trans(k)) * sm_scale          # [BLOCK_M, BLOCK_N]
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -float("inf"))
        if HAS_MASK:
            # atten_mask: 1=不参与 -> 负无穷 (CANN 语义, README L127)
            s = apply_atten_mask(s, pid_b, pid_n, offs_m, offs_n)   # 伪：launcher 提供掩码张量

        # 在线 softmax: 更新 running max/sum，重算本块 P
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])                   # 本块未归一化 P
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        # 步 2: dp = dY @ V^T
        dp = tl.dot(dy, tl.trans(v))                      # [BLOCK_M, BLOCK_N]

        # 步 3: ds = p_norm * (dp - rowsum(dp * p_norm))   (FlashSoftmaxGrad)
        p_norm = p / l_i[:, None]                         # 归一化 P
        if HAS_DROP:
            p_norm = apply_dropout(p_norm, keep_prob)     # 伪：drop_mask 1=保留
        ds = p_norm * (dp - tl.sum(dp * p_norm, axis=1)[:, None])

        # 步 4: dQ += (ds @ K) * scale  (本地累加)
        acc_dq += tl.dot(ds, k) * sm_scale

        # 步 5: dK += (ds^T @ Q) * scale  (跨 query 块 atomic)
        dk_ptrs = dk_ptr + pid_b * stride_dkb + pid_n * stride_dkn + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        tl.atomic_add(dk_ptrs, (tl.dot(tl.trans(ds), q) * sm_scale).to(dk_ptr.dtype.element_ty),
                      mask=mask_n[:, None])

        # 步 6: dV += p_norm^T @ dY  (跨 query 块 atomic)
        dv_ptrs = dv_ptr + pid_b * stride_dvb + pid_n * stride_dvn + offs_n[:, None] * stride_dvs + offs_dv[None, :] * stride_dvd
        tl.atomic_add(dv_ptrs, tl.dot(tl.trans(p_norm), dy).to(dv_ptr.dtype.element_ty),
                      mask=mask_n[:, None])

    # 写回 dQ (步 4)
    dq_ptrs = dq_ptr + pid_b * stride_dqb + pid_n * stride_dqn + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc_dq.to(dq_ptr.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_score_grad_launch(q, k, v, dy, attention_in, sm_scale, keep_prob=1.0,
                                      block_m=128, block_n=128, **kwargs):
    """Launcher：对应 CANN tiling 入口职责——按 shape 选 BLOCK_*、构造 grid、短路空 tensor。"""
    B, N2, G, Sq, D = q.shape  # 简化：实际按 input_layout 解析 (BSH/SBH/BNSD/BSND/TND)
    Skv = k.shape[-2]
    # 空 tensor 短路（CANN FAG_EMPTY_TILING_KEY）：直接零填充 dq/dk/dv，无 matmul
    if Sq == 0 or Skv == 0 or q.numel() == 0:
        return torch_zeros_like(q), torch_zeros_like(k), torch_zeros_like(v)
    grid = (B, N2 * G, triton.cdiv(Sq, block_m))
    _flash_attn_bwd_dq_dk_dv[grid](
        q, k, v, dy, attention_in, dq, dk, dv, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *dy.stride(),
        *dq.stride(), *dk.stride(), *dv.stride(),
        Sq, Skv,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D, DV=v.shape[-1],
        HAS_MASK=kwargs.get("has_mask", False), HAS_PSE=kwargs.get("has_pse", False),
        HAS_DROP=keep_prob < 1.0, IS_CAUSAL=kwargs.get("is_causal", False),
    )
    return dq, dk, dv


# 映射备忘（随 skeleton 记录，供生成器 agent 参考）：
# - program_id(0/1/2) <- CANN 核间切 (B, N2*G, S1_block)
# - BLOCK_M/N <- CANN S1.i/S2.i（arch35: 128/128；arch32 V 侧: 8/1024，需两段 tiling）
# - 尾块 mask <- CANN tail-block/边界处理
# - FP32 累加 <- CANN L0C/UB FP32 中间
# - 重算 P (步1) <- CANN mm1 workspace；Triton 用 on-chip tensor，drop workspace
# - dK/dV atomic <- CANN 多核同址累加（确定性模板用特定分核避免 atomic 非确定性）
# - HAS_MASK/HAS_PSE/HAS_DROP/IS_CAUSAL <- tiling key bit 字段，映射为 constexpr 分支
