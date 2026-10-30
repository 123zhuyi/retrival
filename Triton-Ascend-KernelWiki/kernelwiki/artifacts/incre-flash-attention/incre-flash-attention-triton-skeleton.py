# incre_flash_attention - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：增量解码（query S1=1）FlashAttention 前向，在线 softmax，
#   FlashDecode（split-KV + lse 加权规约）两段式，可选 mask/pse/PA/quant/KV左padding 分支。
# CANN 专有机制（MTE/BufferID/queue/event/ND↔NZ/CV 手动同步/N-Buffer/回调拷贝）不在此体现。
# 来源：attention/incre_flash_attention/ (commit d28d8f9...), docs/IFA算子设计介绍.md L9-173.
#
# 映射注记（见 artifact tiling_dispatch_tables.md §H）：
#   program_id(0) <- CANN core id 沿 BN2 轴
#   program_id(1) <- FlashDecode split-Kv 维（BN2 < 0.4×核数 时使能）
#   BLOCK_M <- S1 = 1（增量解码单 token，设计介绍 L9）
#   BLOCK_N <- S2 基本块（值 source-reported，needs_verification）
#   tail mask <- CANN 尾块/变长 actualSeqLengths 边界
#   FP32 accumulator <- CANN FP32 中间（设计介绍 L59）
#   FlashDecode 两段 <- 设计介绍 L73/119-138（split 计算 + lse 加权 reduce）

import triton
import triton.language as tl


# ========== 阶段 1：FlashDecode split kernel ==========
# 每个 program 处理 (b, n2, split_id)：沿 S2 的一个子段做在线 FlashSoftmax，
# 输出该子段的 partial lse 与 partial o，供 combine 阶段加权规约。
@triton.jit
def ifa_split_kernel(
    # pointers
    q_ptr, k_ptr, v_ptr,
    pse_ptr, atten_mask_ptr,
    block_table_ptr,            # PagedAttention (可选)
    partial_o_ptr,              # [B, N2, num_splits, DV] 输出 partial PV
    partial_lse_ptr,            # [B, N2, num_splits] 输出 partial lse
    # strides (省略，按 layout 计算)
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_ks, stride_kd,
    stride_vb, stride_vn, stride_vs, stride_vd,
    # shape / scalars
    sm_scale,
    actual_kv_len,              # per-batch 实际 KV 长度（actualSeqLengths）
    kv_padding,                 # V4 KV 左 padding（0 = 无）
    # feature flags (constexpr)
    HAS_PSE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    USE_PA: tl.constexpr,       # PagedAttention
    # block meta (constexpr) -- 值 source-reported，非最优声明
    BLOCK_M: tl.constexpr,      # = 1（S1=1 增量解码）
    BLOCK_N: tl.constexpr,      # S2 子段基本块
    D: tl.constexpr,
    DV: tl.constexpr,
    KV_SPLIT: tl.constexpr,     # 每个 split 覆盖的 BLOCK_N 数（FlashDecode 切分数）
):
    pid_bn = tl.program_id(0)   # BN2 维
    pid_split = tl.program_id(1)  # FlashDecode split 维

    # 解析 (b, n2)（简化：假设 BN2 = B * N2 已展平）
    # q 偏移：S1=1，单 token
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + pid_bn * stride_qb_n + offs_d * stride_qd).to(tl.float32)  # [D]

    # KV 左 padding（V4）：起点 = Smax - kv_padding - actual_kv_len
    #   越界（< 0）返回 0（V4 doc L255-263）
    kv_start = kv_padding  # 简化示意；精确 = Smax - kv_padding - actual_kv_len

    # 本 split 负责的 S2 区间
    split_start = pid_split * KV_SPLIT * BLOCK_N
    split_end = tl.minimum(split_start + KV_SPLIT * BLOCK_N, actual_kv_len)

    # 在线 softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)              # PV 累加器

    # S2 子段内 FlashAttention 循环（设计介绍 L22-32）
    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < actual_kv_len

        # ---- PagedAttention：block_table 查物理 KV 偏移（设计介绍 L156-158）----
        if USE_PA:
            kv_offs = _pa_resolve(block_table_ptr, offs_n)   # 伪过程，needs_verification
        else:
            kv_offs = offs_n

        # K 块 [BLOCK_N, D]，V 块 [BLOCK_N, DV]
        k = tl.load(k_ptr + kv_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_offs[:, None] * stride_vs + tl.arange(0, DV)[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)

        # Bmm1: QK^T，FP32 累加（S1=1 -> [BLOCK_N]）
        s = tl.sum(q[:, None] * k, axis=0).to(tl.float32) * sm_scale   # [BLOCK_N]，伪码：1×D · D×BLOCK_N

        # pse（位置编码）add（设计介绍 L20）
        if HAS_PSE:
            pse = tl.load(pse_ptr + offs_n, mask=n_mask, other=0.0)
            s = s + pse

        # atten_mask select -> 负极小（true 位置遮蔽）
        if HAS_MASK:
            mask = tl.load(atten_mask_ptr + offs_n, mask=n_mask, other=0)
            s = tl.where(mask != 0, -float("inf"), s)

        # 在线 FlashSoftmax（设计介绍 L22-32）
        m_ij = tl.max(s, axis=0)                       # 本分块 max
        s = s - m_ij
        p = tl.exp(s).to(v.dtype)                      # exp
        l_ij = tl.sum(p, axis=0)                        # 本分块 sum（FP32）
        alpha = tl.exp(m_i - m_ij)                      # rescale factor exp[i]=e^(max[i-1]-max[i])
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.sum(p[:, None] * v, axis=0)[None, :].to(tl.float32)  # PV rescale + 累加
        m_i = tl.maximum(m_i, m_ij)

    # 输出本 split 的 partial lse 与 partial o（未经 combine 规约）
    partial_lse = tl.log(l_i) + m_i    # needs_verification：精确公式以 vf_softmax_const.h 为准
    tl.store(partial_lse_ptr + pid_bn * num_splits + pid_split, partial_lse)
    tl.store(partial_o_ptr + (pid_bn * num_splits + pid_split) * DV + tl.arange(0, DV),
             (acc / l_i[:, None]).to(partial_o_ptr.dtype.element_ty))


# ========== 阶段 2：FlashDecode combine kernel ==========
# 每个 program 处理 (b, n2)：读所有 split 的 partial lse / partial o，
# lse 加权求和 -> attention_out（设计介绍 L119-138）
@triton.jit
def ifa_combine_kernel(
    partial_o_ptr,              # [B, N2, num_splits, DV]
    partial_lse_ptr,            # [B, N2, num_splits]
    attention_out_ptr,          # [B, N2, DV]（S1=1）
    # shape
    num_splits: tl.constexpr,
    DV: tl.constexpr,
):
    pid_bn = tl.program_id(0)   # BN2 维

    # 读所有 split 的 lse，计算加权系数（设计介绍 L129-135）
    lse = tl.load(partial_lse_ptr + pid_bn * num_splits + tl.arange(0, num_splits))  # [num_splits]
    scale = tl.exp(lse - tl.max(lse, axis=0))
    scale = scale / tl.sum(scale, axis=0)            # exp(lse[i]) / Sum(exp(lse[i]))

    # 加权求和（设计介绍 L133-136）
    res = tl.zeros([DV], dtype=tl.float32)
    for j in range(num_splits):
        split_o = tl.load(partial_o_ptr + (pid_bn * num_splits + j) * DV + tl.arange(0, DV))
        res += split_o.to(tl.float32) * scale[j]

    tl.store(attention_out_ptr + pid_bn * DV + tl.arange(0, DV),
             res.to(attention_out_ptr.dtype.element_ty))


# ========== 非 FlashDecode 路径（单 kernel，split=1 退化） ==========
# 当 BN2 >= 0.4×核数（不使能 FlashDecode）时，单 kernel 沿 S2 全长做在线 FlashSoftmax，
# 直接输出 attention_out（combine 退化为恒等）。逻辑同 ifa_split_kernel 的 S2 全长循环。

# grid_split  = (B * N2, num_splits)     # FlashDecode 使能
# grid_single = (B * N2,)                # FlashDecode 不使能
# grid_combine = (B * N2,)

# ---- 注记分支（不在此展开实现）----
# AntiQuant-MSD（设计介绍 L140-155）：INT8 KV 时，Q 展开成多行 int8，
#   tl.dot(int8 Q, int8 KV, out_dtype=tl.int32) -> reduce，避免对大 B 矩阵反量化。
# MLA（设计介绍 L56）：IFA 不调用 MLA 全量化模板；若支持，Q/K 沿 D 拼接 nope+rope 分段 dot。
# 空 KV：actual_kv_len == 0 -> early return / 输出全零。
