# prompt_flash_attention — Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：prefill 全序列 FlashAttention 前向，在线 softmax，可选 mask/PA/quant/MLA 分支。
# CANN 专有机制（MTE/BufferID/queue/event/ND↔NZ/CV 手动同步）不在此体现。
# 来源：attention/prompt_flash_attention/ (commit d28d8f9...), docs/PFA算子设计介绍.md L9-22.
#
# 映射注记（见 artifact tiling_dispatch_tables.md §H）：
#   program_id(0) <- CANN core id 沿 B/N2/G/S1 轴
#   BLOCK_M <- S1 基本块 128 (CVDIFF_SOUTER_FACTOR_DEFAULT, tiling.cpp L103)
#   BLOCK_N <- S2 基本块 1024 (CVDIFF_SINNER_FACTOR_DEFAULT, tiling.cpp L106)  [值 source-reported]
#   tail mask <- CANN 尾块/边界处理
#   FP32 accumulator <- CANN FP32 中间 (PFA设计介绍 L59)

import triton
import triton.language as tl


@triton.jit
def prompt_flash_attention_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    pse_ptr, atten_mask_ptr,
    block_table_ptr,            # PagedAttention (可选)
    # strides (省略，按 layout 计算)
    stride_qb, stride_qn, stride_qs, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_vb, stride_vn, stride_vs, stride_vd,
    stride_ob, stride_on, stride_os, stride_od,
    # shape / scalars
    sm_scale,
    q_len, kv_len,              # actual_seq_lengths (变长)
    # feature flags (constexpr)
    HAS_PSE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_PA: tl.constexpr,       # PagedAttention
    MLA: tl.constexpr,          # D != Dv 或 D>256
    WRITE_LSE: tl.constexpr,    # 推理输出 lse
    # block meta (constexpr) —— 值 source-reported，非最优声明
    BLOCK_M: tl.constexpr,      # S1 基本块，默认 128
    BLOCK_N: tl.constexpr,      # S2 基本块，默认 1024
    D: tl.constexpr,
    DV: tl.constexpr,
):
    # program 维 <- CANN core 沿 (B, N2, G, S1) 切分；此处用 1D 简化
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    # 变长边界
    m_mask = offs_m < q_len

    # Q 块 [BLOCK_M, D]，FP32 累加
    q = tl.load(q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None], other=0.0).to(tl.float32)

    # 在线 softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)              # PV 累加器

    # S2 轴分块循环（FlashSoftmax 对 Skv 切分，在线刷新）
    for start_n in range(0, kv_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < kv_len

        # ---- PagedAttention：通过 block_table 查物理 KV 偏移（可选分支）----
        if USE_PA:
            # block_id = block_table[batch, offs_n // block_size]
            # kv_phys_off = block_id * block_size + (offs_n % block_size)
            kv_offs = _pa_resolve(block_table_ptr, offs_n)   # 伪过程，需 needs_verification
        else:
            kv_offs = offs_n

        # K 块 [BLOCK_N, D]，V 块 [BLOCK_N, DV]
        k = tl.load(k_ptr + kv_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_offs[:, None] * stride_vs + tl.arange(0, DV)[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)

        # Bmm1: QK^T，FP32 累加
        s = tl.dot(q, k.trans(1, 0)).to(tl.float32) * sm_scale    # [BLOCK_M, BLOCK_N]

        # pse（位置编码）add/mul，顺序由 pse_mode 决定
        if HAS_PSE:
            pse = tl.load(pse_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
                          mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            s = s + pse   # PSE_OUTER_ADD_MUL 示意；其他 pse_mode 需改写

        # atten_mask select → 负极小（true 位置遮蔽）
        if HAS_MASK:
            mask = tl.load(atten_mask_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
                           mask=m_mask[:, None] & n_mask[None, :], other=0)
            s = tl.where(mask != 0, -float("inf"), s)

        # 在线 FlashSoftmax（设计介绍 L13-22）
        m_ij = tl.max(s, axis=1)                       # 本分块行 max
        s = s - m_ij[:, None]
        p = tl.exp(s).to(v.dtype)                      # exp
        l_ij = tl.sum(p, axis=1)                        # 本分块行 sum（FP32）
        alpha = tl.exp(m_i - m_ij)                      # rescale factor exp[i]=e^(max[i-1]-max[i])
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, v).to(tl.float32)   # PV rescale + 累加
        m_i = tl.maximum(m_i, m_ij)

    # rescale 写回（÷softmax_sum，设计介绍 L22）
    acc = acc / l_i[:, None]
    o = acc.to(o_ptr.dtype.element_ty())
    tl.store(o_ptr + offs_m[:, None] * stride_os + tl.arange(0, DV)[None, :] * stride_od,
             o, mask=m_mask[:, None])

    # 推理输出 lse（供增量解码复用）
    if WRITE_LSE:
        lse = tl.log(l_i) + m_i    # needs_verification：精确公式以 vf_softmaxflashv3.h 为准
        tl.store(lse_ptr + offs_m, lse, mask=m_mask)


# MLA 分支（D != Dv 或 D > 256）：Q/K 沿 D 维拼接 nope(512)+rope(64)=576
#   q = [q_nope, q_rope]; k = [k_nope, k_rope]
#   s = tl.dot(q, k.T)  # D=576
#   v 仅用 nope 部分（DV=512）
# 量化分支：FP8/INT8 输入 → cast to FP32 计算 → cast back；deq_scale1/2 调整 scale。
# 空 tensor 分支：q_len==0 → early return；kv_len==0 → 输出全零。

# grid = (ceil(q_len / BLOCK_M) * B * N * G,)   # 多核沿 B/N2/G/S1
